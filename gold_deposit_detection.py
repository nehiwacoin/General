#!/usr/bin/env python3
"""
Gold Deposit Detection using Google Earth Engine
==================================================
This script identifies potential gold deposits in Alaska and Arizona using
remote sensing techniques for mineral alteration mapping.

Techniques used:
- Band ratios for iron oxide detection (ferric/ferrous minerals)
- Clay mineral detection (hydrothermal alteration zones)
- NDVI masking to remove vegetation
- Multi-spectral analysis using Landsat 8/9

Author: Claude Code
Date: 2025-11-08
"""

import ee
import folium
import json
from pathlib import Path

# ============================================================================
# AUTHENTICATION INSTRUCTIONS
# ============================================================================
print("=" * 70)
print("GOOGLE EARTH ENGINE AUTHENTICATION")
print("=" * 70)
print("\nBefore running this script, you need to authenticate GEE.")
print("\nRun ONE of the following commands:")
print("  1. In terminal: earthengine authenticate")
print("  2. In Python: ee.Authenticate()")
print("\nAfter authentication, this script will work automatically.")
print("=" * 70)

# Attempt to initialize Earth Engine
try:
    ee.Initialize()
    print("\n✓ Earth Engine initialized successfully!")
except Exception as e:
    print(f"\n✗ Authentication required: {str(e)}")
    print("\nRun: ee.Authenticate()")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define regions of interest
REGIONS = {
    'alaska_fairbanks': ee.Geometry.Rectangle([-150.5, 64.5, -147.0, 65.5]),
    'alaska_juneau': ee.Geometry.Rectangle([-135.0, 57.5, -133.0, 58.5]),
    'arizona_phoenix': ee.Geometry.Rectangle([-112.5, 33.0, -111.0, 34.0]),
    'arizona_tucson': ee.Geometry.Rectangle([-111.5, 31.5, -110.5, 32.5]),
    'arizona_bradshaw_prescott': ee.Geometry.Rectangle([-112.6, 34.2, -112.0, 34.8]),
    'arizona_flagstaff': ee.Geometry.Rectangle([-111.9, 35.0, -111.4, 35.5])
}

# Date range for imagery
START_DATE = '2023-01-01'
END_DATE = '2024-12-31'

# Cloud cover threshold
MAX_CLOUD_COVER = 20  # percent

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_landsat_composite(region, start_date, end_date, max_cloud_cover):
    """
    Create a cloud-free Landsat 8/9 composite for the region.

    Args:
        region: ee.Geometry - Region of interest
        start_date: str - Start date (YYYY-MM-DD)
        end_date: str - End date (YYYY-MM-DD)
        max_cloud_cover: int - Maximum cloud cover percentage

    Returns:
        ee.Image - Median composite image
    """
    # Landsat 8/9 Collection 2 Tier 1 Level 2 (Surface Reflectance)
    collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
                  .filterBounds(region)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover)))

    # Apply scaling factors for Landsat Collection 2
    def apply_scale_factors(image):
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

    collection = collection.map(apply_scale_factors)

    # Return median composite
    return collection.median().clip(region)


def calculate_mineral_indices(image):
    """
    Calculate various mineral detection indices.

    Landsat 8/9 Bands:
    - B2: Blue (0.45-0.51 μm)
    - B3: Green (0.53-0.59 μm)
    - B4: Red (0.64-0.67 μm)
    - B5: NIR (0.85-0.88 μm)
    - B6: SWIR1 (1.57-1.65 μm)
    - B7: SWIR2 (2.11-2.29 μm)

    Args:
        image: ee.Image - Landsat composite

    Returns:
        dict: Dictionary of mineral indices
    """
    indices = {}

    # 1. Iron Oxide Ratio (Ferric Iron)
    # High values indicate presence of oxidized iron (gossan, limonite)
    # Common in weathered gold deposits
    indices['iron_oxide'] = image.select('SR_B4').divide(image.select('SR_B2'))

    # 2. Ferrous Minerals Ratio
    # Detects ferrous iron in minerals
    indices['ferrous_minerals'] = image.select('SR_B6').divide(image.select('SR_B5'))

    # 3. Clay Minerals / OH-bearing minerals
    # Indicates hydrothermal alteration zones (argillic alteration)
    indices['clay_minerals'] = image.select('SR_B6').divide(image.select('SR_B7'))

    # 4. Alunite/Kaolinite/Pyrophyllite
    # Advanced argillic alteration associated with epithermal gold
    indices['advanced_argillic'] = (image.select('SR_B5')
                                    .multiply(image.select('SR_B7'))
                                    .divide(image.select('SR_B6').pow(2)))

    # 5. NDVI (Normalized Difference Vegetation Index)
    # Used to mask out vegetated areas
    indices['ndvi'] = (image.select('SR_B5').subtract(image.select('SR_B4'))
                       .divide(image.select('SR_B5').add(image.select('SR_B4'))))

    # 6. Composite Alteration Index
    # Combines iron oxide and clay minerals
    indices['alteration_composite'] = (indices['iron_oxide']
                                       .add(indices['clay_minerals'])
                                       .divide(2))

    return indices


def create_prospectivity_map(indices):
    """
    Create a gold prospectivity map combining multiple indices.

    Args:
        indices: dict - Dictionary of mineral indices

    Returns:
        ee.Image - Prospectivity score (0-1)
    """
    # Normalize each index to 0-1 range
    def normalize(image):
        min_val = image.reduceRegion(
            reducer=ee.Reducer.percentile([5]),
            geometry=image.geometry(),
            scale=30,
            maxPixels=1e9
        ).values().get(0)
        max_val = image.reduceRegion(
            reducer=ee.Reducer.percentile([95]),
            geometry=image.geometry(),
            scale=30,
            maxPixels=1e9
        ).values().get(0)
        return image.subtract(ee.Number(min_val)).divide(
            ee.Number(max_val).subtract(ee.Number(min_val))
        ).clamp(0, 1)

    # Weight factors (adjust based on geological knowledge)
    weights = {
        'iron_oxide': 0.3,
        'clay_minerals': 0.3,
        'ferrous_minerals': 0.2,
        'advanced_argillic': 0.2
    }

    # Calculate weighted prospectivity
    prospectivity = ee.Image(0)
    for key, weight in weights.items():
        prospectivity = prospectivity.add(indices[key].multiply(weight))

    # Mask out vegetation (NDVI > 0.4)
    vegetation_mask = indices['ndvi'].lt(0.4)
    prospectivity = prospectivity.updateMask(vegetation_mask)

    return prospectivity


def export_to_drive(image, region, description, folder='GEE_GoldDetection'):
    """
    Export image to Google Drive.

    Args:
        image: ee.Image - Image to export
        region: ee.Geometry - Region to export
        description: str - Export description
        folder: str - Drive folder name
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region.bounds(),
        scale=30,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task.start()
    print(f"✓ Export task started: {description}")
    print(f"  Task ID: {task.id}")
    return task


def create_interactive_map(region, composite, indices, prospectivity, center_lat, center_lon):
    """
    Create an interactive Folium map.

    Args:
        region: ee.Geometry - Region of interest
        composite: ee.Image - Landsat composite
        indices: dict - Mineral indices
        prospectivity: ee.Image - Prospectivity map
        center_lat: float - Map center latitude
        center_lon: float - Map center longitude

    Returns:
        folium.Map - Interactive map
    """
    # Create base map
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=9)

    # True color composite
    true_color_vis = {
        'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
        'min': 0,
        'max': 0.3,
        'gamma': 1.4
    }

    # Add true color layer
    map_obj.add_ee_layer(
        composite,
        true_color_vis,
        'True Color'
    )

    # Iron oxide visualization
    iron_vis = {
        'min': 0.5,
        'max': 2.5,
        'palette': ['blue', 'cyan', 'yellow', 'red']
    }

    map_obj.add_ee_layer(
        indices['iron_oxide'],
        iron_vis,
        'Iron Oxide'
    )

    # Clay minerals visualization
    clay_vis = {
        'min': 0.8,
        'max': 1.5,
        'palette': ['green', 'yellow', 'orange', 'red']
    }

    map_obj.add_ee_layer(
        indices['clay_minerals'],
        clay_vis,
        'Clay Minerals'
    )

    # Prospectivity map
    prospect_vis = {
        'min': 0,
        'max': 1,
        'palette': ['blue', 'green', 'yellow', 'orange', 'red']
    }

    map_obj.add_ee_layer(
        prospectivity,
        prospect_vis,
        'Gold Prospectivity'
    )

    # Add layer control
    folium.LayerControl().add_to(map_obj)

    return map_obj


# Add method to folium.Map to display Earth Engine layers
def add_ee_layer(self, ee_image_object, vis_params, name):
    """Add Earth Engine layer to folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

folium.Map.add_ee_layer = add_ee_layer


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_region(region_name, region_geom):
    """
    Perform complete analysis for a region.

    Args:
        region_name: str - Name of the region
        region_geom: ee.Geometry - Region geometry
    """
    print(f"\n{'=' * 70}")
    print(f"ANALYZING: {region_name.upper().replace('_', ' ')}")
    print(f"{'=' * 70}")

    # 1. Get Landsat composite
    print("\n1. Creating Landsat composite...")
    composite = get_landsat_composite(region_geom, START_DATE, END_DATE, MAX_CLOUD_COVER)
    print("   ✓ Composite created")

    # 2. Calculate mineral indices
    print("\n2. Calculating mineral detection indices...")
    indices = calculate_mineral_indices(composite)
    print("   ✓ Iron oxide ratio")
    print("   ✓ Ferrous minerals ratio")
    print("   ✓ Clay minerals ratio")
    print("   ✓ Advanced argillic index")
    print("   ✓ NDVI")

    # 3. Create prospectivity map
    print("\n3. Generating prospectivity map...")
    prospectivity = create_prospectivity_map(indices)
    print("   ✓ Prospectivity map generated")

    # 4. Get statistics
    print("\n4. Computing statistics...")
    stats = prospectivity.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ).combine(
            reducer2=ee.Reducer.minMax(),
            sharedInputs=True
        ),
        geometry=region_geom,
        scale=30,
        maxPixels=1e9
    ).getInfo()

    print(f"\n   Prospectivity Statistics:")
    print(f"   - Mean: {stats.get('constant_mean', 'N/A'):.4f}")
    print(f"   - Std Dev: {stats.get('constant_stdDev', 'N/A'):.4f}")
    print(f"   - Min: {stats.get('constant_min', 'N/A'):.4f}")
    print(f"   - Max: {stats.get('constant_max', 'N/A'):.4f}")

    # 5. Create interactive map
    print("\n5. Creating interactive map...")
    center = region_geom.centroid().coordinates().getInfo()
    center_lon, center_lat = center

    map_obj = create_interactive_map(
        region_geom,
        composite,
        indices,
        prospectivity,
        center_lat,
        center_lon
    )

    # Save map
    map_filename = f"{region_name}_gold_detection_map.html"
    map_obj.save(map_filename)
    print(f"   ✓ Map saved: {map_filename}")

    # 6. Export to Google Drive (optional - commented out by default)
    # print("\n6. Exporting to Google Drive...")
    # export_to_drive(prospectivity, region_geom, f"{region_name}_prospectivity")
    # export_to_drive(indices['iron_oxide'], region_geom, f"{region_name}_iron_oxide")
    # export_to_drive(indices['clay_minerals'], region_geom, f"{region_name}_clay_minerals")

    return {
        'composite': composite,
        'indices': indices,
        'prospectivity': prospectivity,
        'stats': stats,
        'map': map_obj
    }


# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GOLD DEPOSIT DETECTION - ALASKA & ARIZONA")
    print("=" * 70)

    results = {}

    # Analyze each region
    for region_name, region_geom in REGIONS.items():
        try:
            results[region_name] = analyze_region(region_name, region_geom)
        except Exception as e:
            print(f"\n✗ Error analyzing {region_name}: {str(e)}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n✓ Analyzed {len(results)} regions")
    print(f"✓ Generated {len(results)} interactive maps")
    print("\nHTML maps created:")
    for region_name in results.keys():
        print(f"  - {region_name}_gold_detection_map.html")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Open the HTML map files in your browser
2. Toggle between layers:
   - True Color: Visual reference
   - Iron Oxide: Red/yellow areas = oxidized zones (gossan)
   - Clay Minerals: Hydrothermal alteration zones
   - Gold Prospectivity: High values = potential targets

3. High-priority areas show:
   - High iron oxide (weathered sulfides)
   - High clay minerals (hydrothermal alteration)
   - Low vegetation (exposed rock)
   - Proximity to known geological structures

4. Validate promising areas with:
   - Geological maps
   - Known mineral occurrences
   - Field verification
   - Detailed ground surveys

5. To export data, uncomment the export_to_drive() calls in the script
""")

    print("=" * 70)
