#!/usr/bin/env python3
"""
Mat-Su Valley Lode Gold Prospectivity Mapping - Version 2.0
============================================================

This version addresses the "everything is red" problem with:
- Proper percentile-based normalization
- Willow-specific spectral indices (emphasizing carbonate alteration)
- Structural analysis (fault/lineament detection)
- Validation against known mine locations
- Machine learning classification (Random Forest)

Author: Claude Code
Date: 2025-11-09
Version: 2.0.0
"""

import ee
import folium
import pandas as pd
from pathlib import Path
import json

# Import our modules
import config
import utils
from validation import ValidationFramework

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize():
    """Initialize Earth Engine and print configuration."""
    print("\n" + "=" * 70)
    print("MAT-SU VALLEY LODE GOLD PROSPECTIVITY MAPPING v2.0")
    print("=" * 70)

    # Initialize Earth Engine
    if not config.initialize_earth_engine():
        print("\n✗ Earth Engine initialization failed")
        print("Please authenticate first: python -c \"import ee; ee.Authenticate()\"")
        exit(1)

    # Print configuration
    config.print_config_summary()


# ============================================================================
# PHASE 1: DATA ACQUISITION
# ============================================================================

def acquire_base_data(region):
    """
    Acquire base datasets: Landsat composite and DEM.

    Args:
        region: ee.Geometry - Study area

    Returns:
        dict with 'composite' and 'dem'
    """
    print("\n" + "=" * 70)
    print("PHASE 1: DATA ACQUISITION")
    print("=" * 70)

    # Get optimized Landsat composite
    composite = utils.get_optimized_landsat_composite(
        region,
        config.PRIMARY_DATE_RANGE['start'],
        config.PRIMARY_DATE_RANGE['end']
    )

    # Get DEM
    dem = utils.get_dem_data(region, dem_type='alos')

    print("\n✓ Phase 1 complete: Base data acquired")

    return {
        'composite': composite,
        'dem': dem
    }


# ============================================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================================

def calculate_all_indices(composite, dem, region):
    """
    Calculate all spectral and structural indices.

    Args:
        composite: ee.Image - Landsat composite
        dem: ee.Image - Digital elevation model
        region: ee.Geometry - Study area

    Returns:
        dict with 'spectral' and 'structural' indices
    """
    print("\n" + "=" * 70)
    print("PHASE 2: FEATURE ENGINEERING")
    print("=" * 70)

    # Spectral indices (with proper normalization)
    spectral_indices = utils.calculate_willow_spectral_indices(composite, region)

    # Terrain indices
    terrain_indices = utils.calculate_terrain_indices(dem)

    # Lineament detection
    lineaments = utils.detect_lineaments_multidirectional(dem, region)

    # Fault proximity
    fault_proximity = utils.calculate_fault_proximity(lineaments, region)

    # Combine structural indices
    structural_indices = {
        'slope': terrain_indices['slope'],
        'curvature': terrain_indices['curvature'],
        'lineaments': lineaments,
        'fault_proximity': fault_proximity,
    }

    print("\n✓ Phase 2 complete: Feature engineering done")

    return {
        'spectral': spectral_indices,
        'structural': structural_indices
    }


# ============================================================================
# PHASE 3: PROSPECTIVITY MODELING (WITHOUT ML - SIMPLIFIED)
# ============================================================================

def create_prospectivity_model_simple(spectral_indices, structural_indices, region):
    """
    Create prospectivity model using weighted combination (no ML for now).

    This is a simplified version that doesn't require ML training.
    We'll add ML in Phase 3 after validation framework is tested.

    Args:
        spectral_indices: dict - Spectral indices
        structural_indices: dict - Structural indices
        region: ee.Geometry - Study area

    Returns:
        ee.Image - Prospectivity map (0-1 scale)
    """
    print("\n" + "=" * 70)
    print("PHASE 3: PROSPECTIVITY MODELING (Simplified)")
    print("=" * 70)

    # Use the Willow signature (already weighted)
    spectral_component = spectral_indices['willow_signature']

    # Combine structural features
    structural_component = (
        structural_indices['fault_proximity'].multiply(0.6)  # Proximity is key
        .add(structural_indices['lineaments'].multiply(0.3))
        .add(structural_indices['curvature'].divide(100).clamp(0, 1).multiply(0.1))
    )

    # Overall prospectivity (weighted combination)
    weights = config.PROSPECTIVITY_WEIGHTS
    prospectivity = (
        spectral_component.multiply(weights['spectral_alteration'])
        .add(structural_component.multiply(weights['structural']))
    )

    # Note: ML component weight distributed to spectral/structural for now
    # Normalize weights: 0.30 + 0.45 = 0.75 base, add 0.25 distributed
    prospectivity = prospectivity.divide(0.75).clamp(0, 1)

    # Mask out vegetation (NDVI > 0.4)
    vegetation_mask = spectral_indices['ndvi'].lt(0.4)
    prospectivity = prospectivity.updateMask(vegetation_mask).rename('prospectivity')

    print("  ✓ Prospectivity model created")
    print(f"    - Spectral weight: {weights['spectral_alteration']}")
    print(f"    - Structural weight: {weights['structural']}")
    print(f"    - ML weight: {weights['ml_probability']} (not yet implemented)")

    print("\n✓ Phase 3 complete: Prospectivity model created")

    return prospectivity


# ============================================================================
# PHASE 4: VALIDATION
# ============================================================================

def validate_prospectivity(prospectivity, user_mines_csv=None):
    """
    Validate prospectivity model against known mines.

    Args:
        prospectivity: ee.Image - Prospectivity map
        user_mines_csv: str - Optional path to user's mine CSV

    Returns:
        dict - Validation metrics
    """
    print("\n" + "=" * 70)
    print("PHASE 4: VALIDATION")
    print("=" * 70)

    validator = ValidationFramework(user_mines_csv)
    metrics = validator.validate_model(prospectivity)

    if not metrics['passed']:
        validator.suggest_improvements()

    print("\n✓ Phase 4 complete: Validation done")

    return metrics


# ============================================================================
# PHASE 5: OUTPUT GENERATION
# ============================================================================

def create_interactive_map(region, composite, spectral_indices, structural_indices,
                          prospectivity, center_lat, center_lon):
    """
    Create interactive Folium map with all layers.

    Args:
        region: ee.Geometry
        composite: ee.Image
        spectral_indices: dict
        structural_indices: dict
        prospectivity: ee.Image
        center_lat: float
        center_lon: float

    Returns:
        folium.Map
    """
    print("\nCreating interactive map...")

    # Create base map
    map_obj = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    # Helper function to add EE layer
    def add_ee_layer(ee_object, vis_params, name, show=True):
        map_id = ee.Image(ee_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name=name,
            overlay=True,
            control=True,
            show=show
        ).add_to(map_obj)

    # Add layers
    print("  Adding layers...")

    # 1. True color
    add_ee_layer(
        composite,
        config.MAP_VIS_PARAMS['true_color'],
        '1. True Color (Landsat)',
        show=False
    )

    # 2. Iron oxide
    add_ee_layer(
        spectral_indices['iron_oxide'],
        config.MAP_VIS_PARAMS['iron_oxide'],
        '2. Iron Oxide (Gossan)',
        show=False
    )

    # 3. Carbonate (DIAGNOSTIC)
    add_ee_layer(
        spectral_indices['carbonate'],
        config.MAP_VIS_PARAMS['carbonate'],
        '3. Carbonate Alteration (DIAGNOSTIC)',
        show=False
    )

    # 4. Sericite
    add_ee_layer(
        spectral_indices['sericite'],
        config.MAP_VIS_PARAMS['iron_oxide'],
        '4. Sericite (Phyllic Alteration)',
        show=False
    )

    # 5. Fault proximity
    add_ee_layer(
        structural_indices['fault_proximity'],
        config.MAP_VIS_PARAMS['iron_oxide'],
        '5. Fault Proximity',
        show=False
    )

    # 6. Lineaments
    add_ee_layer(
        structural_indices['lineaments'],
        config.MAP_VIS_PARAMS['iron_oxide'],
        '6. Lineaments (Faults)',
        show=False
    )

    # 7. PROSPECTIVITY (show by default)
    add_ee_layer(
        prospectivity,
        config.MAP_VIS_PARAMS['prospectivity'],
        '7. GOLD PROSPECTIVITY',
        show=True
    )

    # Add known mines as markers
    for mine_name, mine_info in config.KNOWN_MINES.items():
        folium.Marker(
            location=[mine_info['lat'], mine_info['lon']],
            popup=f"<b>{mine_name}</b><br>Type: {mine_info['type']}",
            tooltip=mine_name,
            icon=folium.Icon(color='red', icon='star')
        ).add_to(map_obj)

    # Add layer control
    folium.LayerControl(position='topright', collapsed=False).add_to(map_obj)

    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 250px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <h4>Gold Prospectivity</h4>
    <p><span style="background-color: #a50026; width: 20px; height: 10px; display: inline-block;"></span> Very High (>0.70)</p>
    <p><span style="background-color: #f46d43; width: 20px; height: 10px; display: inline-block;"></span> High (0.50-0.70)</p>
    <p><span style="background-color: #ffffbf; width: 20px; height: 10px; display: inline-block;"></span> Medium (0.30-0.50)</p>
    <p><span style="background-color: #74add1; width: 20px; height: 10px; display: inline-block;"></span> Low (<0.30)</p>
    <p><b>Red stars:</b> Known mines</p>
    </div>
    '''
    map_obj.get_root().html.add_child(folium.Element(legend_html))

    print("  ✓ Map created with 7 layers")

    return map_obj


def generate_outputs(region, composite, spectral_indices, structural_indices,
                    prospectivity, validation_metrics):
    """
    Generate all output files.

    Args:
        region: ee.Geometry
        composite: ee.Image
        spectral_indices: dict
        structural_indices: dict
        prospectivity: ee.Image
        validation_metrics: dict

    Returns:
        dict - Paths to generated files
    """
    print("\n" + "=" * 70)
    print("PHASE 5: OUTPUT GENERATION")
    print("=" * 70)

    output_files = {}

    # Get center coordinates
    center = region.centroid().coordinates().getInfo()
    center_lon, center_lat = center

    # 1. Interactive map
    print("\n1. Creating interactive map...")
    map_obj = create_interactive_map(
        region, composite, spectral_indices, structural_indices,
        prospectivity, center_lat, center_lon
    )

    map_path = config.MAPS_DIR / f"{config.OUTPUT_PREFIX}_prospectivity_map_v2.html"
    map_obj.save(str(map_path))
    output_files['map'] = map_path
    print(f"  ✓ Map saved: {map_path}")

    # 2. Extract high-priority targets
    print("\n2. Extracting high-priority targets...")

    # Threshold for high prospectivity
    high_threshold = config.PROSPECTIVITY_THRESHOLDS['high']
    high_areas = prospectivity.gt(high_threshold)

    # Convert to vectors (simplified - just get sample points)
    sample_points = prospectivity.sample(
        region=region,
        scale=30,
        numPixels=1000,
        seed=42,
        geometries=True
    )

    # Get as feature collection
    high_points = sample_points.filter(
        ee.Filter.gte('prospectivity', high_threshold)
    )

    try:
        targets_data = high_points.getInfo()
        targets_list = []

        for feature in targets_data['features']:
            coords = feature['geometry']['coordinates']
            props = feature['properties']
            targets_list.append({
                'longitude': coords[0],
                'latitude': coords[1],
                'prospectivity': props.get('prospectivity', 0),
                'priority': 'HIGH' if props.get('prospectivity', 0) >= 0.80 else 'MEDIUM'
            })

        # Sort by prospectivity
        targets_list.sort(key=lambda x: x['prospectivity'], reverse=True)

        # Save as CSV
        targets_df = pd.DataFrame(targets_list)
        targets_csv = config.EXPORTS_DIR / f"{config.OUTPUT_PREFIX}_targets.csv"
        targets_df.to_csv(targets_csv, index=False)
        output_files['targets_csv'] = targets_csv
        print(f"  ✓ Targets CSV saved: {targets_csv}")
        print(f"    - {len(targets_list)} high-priority targets")

        # Save as KML
        targets_kml = config.EXPORTS_DIR / f"{config.OUTPUT_PREFIX}_targets.kml"
        kml_content = create_kml(targets_list)
        with open(targets_kml, 'w') as f:
            f.write(kml_content)
        output_files['targets_kml'] = targets_kml
        print(f"  ✓ Targets KML saved: {targets_kml}")

    except Exception as e:
        print(f"  ⚠ Could not extract targets: {e}")

    # 3. Summary report
    print("\n3. Creating summary report...")

    report = generate_summary_report(validation_metrics, output_files)
    report_path = config.OUTPUT_DIR / f"{config.OUTPUT_PREFIX}_report.txt"

    with open(report_path, 'w') as f:
        f.write(report)

    output_files['report'] = report_path
    print(f"  ✓ Report saved: {report_path}")

    print("\n✓ Phase 5 complete: Outputs generated")

    return output_files


def create_kml(targets_list):
    """Create KML file content for targets."""
    kml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Mat-Su Gold Prospectivity Targets</name>
  <Style id="high">
    <IconStyle>
      <color>ff0000ff</color>
      <scale>1.2</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/star.png</href></Icon>
    </IconStyle>
  </Style>
  <Style id="medium">
    <IconStyle>
      <color>ff00ffff</color>
      <scale>1.0</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/star.png</href></Icon>
    </IconStyle>
  </Style>
'''

    placemarks = []
    for i, target in enumerate(targets_list[:100], 1):  # Top 100
        placemark = f'''  <Placemark>
    <name>Target {i}</name>
    <description>Prospectivity: {target['prospectivity']:.3f}&#10;Priority: {target['priority']}</description>
    <styleUrl>#{target['priority'].lower()}</styleUrl>
    <Point>
      <coordinates>{target['longitude']},{target['latitude']},0</coordinates>
    </Point>
  </Placemark>
'''
        placemarks.append(placemark)

    kml_footer = '''</Document>
</kml>'''

    return kml_header + '\n'.join(placemarks) + kml_footer


def generate_summary_report(validation_metrics, output_files):
    """Generate summary report text."""
    report = f"""
{'=' * 70}
MAT-SU VALLEY LODE GOLD PROSPECTIVITY ANALYSIS
Version 2.0
{'=' * 70}

STUDY AREA: {config.STUDY_AREA}
DEPOSIT TYPE: {config.DEPOSIT_TYPE}
DATE RANGE: {config.PRIMARY_DATE_RANGE['start']} to {config.PRIMARY_DATE_RANGE['end']}

{'=' * 70}
VALIDATION RESULTS
{'=' * 70}

Total Known Mines: {validation_metrics['total_mines']}
Valid Samples: {validation_metrics['valid_samples']}

Classification:
  HIGH:      {validation_metrics['high_count']} ({validation_metrics['high_count']/validation_metrics['valid_samples']*100:.1f}%)
  MEDIUM:    {validation_metrics['medium_count']} ({validation_metrics['medium_count']/validation_metrics['valid_samples']*100:.1f}%)
  LOW:       {validation_metrics['low_count']} ({validation_metrics['low_count']/validation_metrics['valid_samples']*100:.1f}%)
  VERY_LOW:  {validation_metrics['very_low_count']} ({validation_metrics['very_low_count']/validation_metrics['valid_samples']*100:.1f}%)

Detection Rate: {validation_metrics['detection_rate']*100:.1f}%
Required Rate:  {validation_metrics['required_rate']*100:.1f}%

STATUS: {'PASSED ✓' if validation_metrics['passed'] else 'FAILED ✗'}

{'=' * 70}
OUTPUT FILES
{'=' * 70}

"""
    for file_type, file_path in output_files.items():
        report += f"{file_type.upper()}: {file_path}\n"

    report += f"""
{'=' * 70}
INTERPRETATION GUIDE
{'=' * 70}

HIGH PROSPECTIVITY ZONES (Red/Orange on map):
- Strong spectral signatures: iron oxide + carbonate alteration
- Proximity to structural features (faults, lineaments)
- Similar signatures to known producing mines
- PRIORITY: Field verification recommended

MEDIUM PROSPECTIVITY (Yellow):
- Moderate alteration signatures
- Some structural control
- Worth investigating if near access

LOW PROSPECTIVITY (Blue/Green):
- Weak or absent alteration
- Lacks structural control
- Lower priority

KNOWN MINES (Red stars on map):
- Historical producing mines in Willow Creek district
- Used for model validation
- {validation_metrics['high_count']} out of {validation_metrics['valid_samples']} detected in HIGH zones

{'=' * 70}
NEXT STEPS
{'=' * 70}

1. Open the HTML map in your browser
2. Toggle between layers to understand each component
3. Review high-priority targets in CSV/KML files
4. Cross-reference with geological maps
5. Plan field verification of top targets
6. If validation failed, adjust weights in config.py

{'=' * 70}
"""

    return report


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main(area_type='core', user_mines_csv=None):
    """
    Main workflow for Mat-Su gold prospectivity analysis.

    Args:
        area_type: str - 'core' or 'extended'
        user_mines_csv: str - Optional path to user's mine validation CSV
    """
    # Initialize
    initialize()

    # Get study area
    region = config.get_study_area_geometry(area_type)

    # Phase 1: Acquire base data
    base_data = acquire_base_data(region)

    # Phase 2: Calculate indices
    indices = calculate_all_indices(
        base_data['composite'],
        base_data['dem'],
        region
    )

    # Phase 3: Create prospectivity model
    prospectivity = create_prospectivity_model_simple(
        indices['spectral'],
        indices['structural'],
        region
    )

    # Phase 4: Validate
    validation_metrics = validate_prospectivity(prospectivity, user_mines_csv)

    # Phase 5: Generate outputs
    output_files = generate_outputs(
        region,
        base_data['composite'],
        indices['spectral'],
        indices['structural'],
        prospectivity,
        validation_metrics
    )

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nValidation: {'✓ PASSED' if validation_metrics['passed'] else '✗ FAILED'}")
    print(f"Detection Rate: {validation_metrics['detection_rate']*100:.1f}%")
    print(f"\nOutput files:")
    for file_type, file_path in output_files.items():
        print(f"  - {file_path}")
    print("\n" + "=" * 70)

    return {
        'prospectivity': prospectivity,
        'validation': validation_metrics,
        'outputs': output_files
    }


if __name__ == "__main__":
    import sys

    # Check for user mines CSV argument
    user_csv = None
    if len(sys.argv) > 1:
        user_csv = sys.argv[1]
        print(f"\nUsing user mines CSV: {user_csv}")

    # Run analysis
    results = main(area_type='core', user_mines_csv=user_csv)

    print("\n" + "=" * 70)
    print("To use your own mine validation data:")
    print("  python matsu_gold_detection_v2.py validation/user_mines_matsu.csv")
    print("=" * 70)
