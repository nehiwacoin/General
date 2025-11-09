"""
Utility Functions for Mat-Su Gold Prospectivity System
=======================================================
Common helper functions for image processing, normalization, and analysis.
"""

import ee
import config

# ============================================================================
# IMAGE ACQUISITION & PREPROCESSING
# ============================================================================

def get_optimized_landsat_composite(region, start_date, end_date):
    """
    Create optimized Landsat composite with seasonal filtering and quality control.

    This addresses the "everything is red" problem by:
    - Using summer months only (snow-free)
    - Stricter cloud/snow filtering
    - Better quality pixel selection

    Args:
        region: ee.Geometry - Region of interest
        start_date: str - Start date (YYYY-MM-DD)
        end_date: str - End date (YYYY-MM-DD)

    Returns:
        ee.Image - Optimized composite
    """
    print("\nCreating optimized Landsat composite...")

    # Merge Landsat 8 and 9 collections
    collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
                  .filterBounds(region)
                  .filterDate(start_date, end_date))

    # Filter by cloud cover
    collection = collection.filter(
        ee.Filter.lt('CLOUD_COVER', config.MAX_CLOUD_COVER)
    )

    # Filter to summer months (June-September for Alaska)
    collection = collection.filter(
        ee.Filter.calendarRange(
            config.SUMMER_MONTHS[0],
            config.SUMMER_MONTHS[-1],
            'month'
        )
    )

    print(f"  Available images: {collection.size().getInfo()}")

    # Apply scaling factors
    def apply_scale_factors(image):
        """Apply Landsat Collection 2 scale factors."""
        optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(optical, None, True).addBands(thermal, None, True)

    # Cloud and snow masking
    def mask_clouds_and_snow(image):
        """Mask clouds, cloud shadows, and snow using QA_PIXEL band."""
        qa = image.select('QA_PIXEL')

        # Bit masks
        cloud_bit = 1 << 3
        cloud_shadow_bit = 1 << 4
        snow_bit = 1 << 5

        # Create masks
        cloud_mask = qa.bitwiseAnd(cloud_bit).eq(0)
        shadow_mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0)
        snow_mask = qa.bitwiseAnd(snow_bit).eq(0)

        # Combine masks
        mask = cloud_mask.And(shadow_mask).And(snow_mask)

        return image.updateMask(mask)

    # Apply processing
    collection = (collection
                  .map(apply_scale_factors)
                  .map(mask_clouds_and_snow))

    # Create median composite
    composite = collection.median().clip(region)

    print("  ✓ Composite created")

    return composite


def get_dem_data(region, dem_type='alos'):
    """
    Get Digital Elevation Model data.

    Args:
        region: ee.Geometry - Region of interest
        dem_type: str - 'alos' (preferred) or 'srtm'

    Returns:
        ee.Image - DEM (single band: 'elevation')
    """
    print(f"\nAcquiring {dem_type.upper()} DEM data...")

    if dem_type == 'alos':
        dem = ee.Image('JAXA/ALOS/AW3D30/V3_2').select('DSM').rename('elevation')
    elif dem_type == 'srtm':
        dem = ee.Image('USGS/SRTMGL1_003').select('elevation')
    else:
        raise ValueError(f"Unknown DEM type: {dem_type}")

    dem = dem.clip(region)
    print("  ✓ DEM acquired")

    return dem


# ============================================================================
# NORMALIZATION FUNCTIONS - FIXES "EVERYTHING IS RED" PROBLEM
# ============================================================================

def normalize_index_percentile(index_image, region, percentile_low=5, percentile_high=95):
    """
    Robust normalization using percentiles.

    THIS FIXES THE "EVERYTHING IS RED" PROBLEM!

    Instead of using min/max (sensitive to outliers), this uses the 5th and 95th
    percentiles to normalize values. This ensures only the top ~5% of values
    show as "high" prospectivity.

    Args:
        index_image: ee.Image - Index to normalize
        region: ee.Geometry - Region for statistics
        percentile_low: int - Lower percentile (default 5)
        percentile_high: int - Upper percentile (default 95)

    Returns:
        ee.Image - Normalized index (0-1 range)
    """
    # Get band name
    band_name = index_image.bandNames().get(0)

    # Calculate percentiles
    stats = index_image.reduceRegion(
        reducer=ee.Reducer.percentile([percentile_low, percentile_high]),
        geometry=region,
        scale=30,
        maxPixels=1e9,
        bestEffort=True
    )

    # Extract percentile values
    p_low = ee.Number(stats.get(ee.String(band_name).cat('_p').cat(str(percentile_low))))
    p_high = ee.Number(stats.get(ee.String(band_name).cat('_p').cat(str(percentile_high))))

    # Normalize to 0-1 range
    normalized = (index_image.subtract(p_low)
                  .divide(p_high.subtract(p_low))
                  .clamp(0, 1)
                  .rename(band_name))

    return normalized


def normalize_index_zscore(index_image, region):
    """
    Z-score normalization.

    Args:
        index_image: ee.Image - Index to normalize
        region: ee.Geometry - Region for statistics

    Returns:
        ee.Image - Z-score normalized index
    """
    band_name = index_image.bandNames().get(0)

    # Calculate mean and std dev
    stats = index_image.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ),
        geometry=region,
        scale=30,
        maxPixels=1e9,
        bestEffort=True
    )

    mean = ee.Number(stats.get(ee.String(band_name).cat('_mean')))
    std = ee.Number(stats.get(ee.String(band_name).cat('_stdDev')))

    # Z-score: (x - mean) / std
    normalized = (index_image.subtract(mean)
                  .divide(std)
                  .rename(band_name))

    return normalized


def auto_normalize(index_image, region):
    """
    Automatically normalize index using configured method.

    Args:
        index_image: ee.Image - Index to normalize
        region: ee.Geometry - Region for statistics

    Returns:
        ee.Image - Normalized index
    """
    method = config.NORMALIZATION_CONFIG['method']

    if method == 'percentile':
        return normalize_index_percentile(
            index_image,
            region,
            config.NORMALIZATION_CONFIG['percentile_low'],
            config.NORMALIZATION_CONFIG['percentile_high']
        )
    elif method == 'z-score':
        return normalize_index_zscore(index_image, region)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ============================================================================
# SPECTRAL INDICES - WILLOW-SPECIFIC
# ============================================================================

def calculate_willow_spectral_indices(image, region):
    """
    Calculate Willow Creek-specific spectral indices.

    These indices emphasize the diagnostic alteration minerals for
    orogenic lode gold deposits:
    - Iron oxide (gossan)
    - Carbonate (ankerite/calcite) - DIAGNOSTIC for Willow
    - Sericite (phyllic alteration)
    - Chlorite (propylitic alteration)

    Args:
        image: ee.Image - Landsat composite
        region: ee.Geometry - Study area for normalization

    Returns:
        dict - Dictionary of normalized spectral indices
    """
    print("\nCalculating Willow-specific spectral indices...")

    indices = {}

    # 1. Iron Oxide (Ferric Iron) - Red/Blue ratio
    # High values = oxidized iron (gossan, limonite)
    iron_oxide = image.select('SR_B4').divide(image.select('SR_B2')).rename('iron_oxide')
    indices['iron_oxide'] = auto_normalize(iron_oxide, region)
    print("  ✓ Iron oxide index")

    # 2. Carbonate (Ankerite/Calcite) - DIAGNOSTIC for Willow deposits
    # SWIR2/SWIR1 ratio
    carbonate = image.select('SR_B7').divide(image.select('SR_B6')).rename('carbonate')
    indices['carbonate'] = auto_normalize(carbonate, region)
    print("  ✓ Carbonate index (DIAGNOSTIC)")

    # 3. Sericite/Muscovite (Phyllic Alteration)
    # (NIR * SWIR2) / SWIR1^2
    sericite = (image.select('SR_B5').multiply(image.select('SR_B7'))
                .divide(image.select('SR_B6').pow(2))
                .rename('sericite'))
    indices['sericite'] = auto_normalize(sericite, region)
    print("  ✓ Sericite index")

    # 4. Chlorite (Propylitic Alteration)
    # SWIR1/NIR ratio
    chlorite = image.select('SR_B6').divide(image.select('SR_B5')).rename('chlorite')
    indices['chlorite'] = auto_normalize(chlorite, region)
    print("  ✓ Chlorite index")

    # 5. Ferrous Minerals
    # SWIR1/NIR ratio (different interpretation than chlorite)
    ferrous = image.select('SR_B6').divide(image.select('SR_B5')).rename('ferrous')
    indices['ferrous'] = auto_normalize(ferrous, region)
    print("  ✓ Ferrous minerals index")

    # 6. NDVI (for masking vegetation)
    ndvi = (image.select('SR_B5').subtract(image.select('SR_B4'))
            .divide(image.select('SR_B5').add(image.select('SR_B4')))
            .rename('ndvi'))
    indices['ndvi'] = ndvi  # Don't normalize NDVI, keep -1 to 1 range
    print("  ✓ NDVI")

    # 7. Willow Composite Signature
    # Weighted combination emphasizing carbonate (40%)
    willow_signature = (
        indices['iron_oxide'].multiply(config.WILLOW_INDEX_WEIGHTS['iron_oxide'])
        .add(indices['carbonate'].multiply(config.WILLOW_INDEX_WEIGHTS['carbonate']))
        .add(indices['sericite'].multiply(config.WILLOW_INDEX_WEIGHTS['sericite']))
        .add(indices['chlorite'].multiply(config.WILLOW_INDEX_WEIGHTS['chlorite']))
        .rename('willow_signature')
    )
    indices['willow_signature'] = willow_signature
    print("  ✓ Willow composite signature")

    return indices


# ============================================================================
# STRUCTURAL ANALYSIS
# ============================================================================

def calculate_terrain_indices(dem):
    """
    Calculate terrain-derived indices for structural analysis.

    Args:
        dem: ee.Image - Digital elevation model

    Returns:
        dict - Terrain indices (slope, aspect, hillshade, etc.)
    """
    print("\nCalculating terrain indices...")

    terrain = ee.Algorithms.Terrain(dem)

    indices = {
        'slope': terrain.select('slope'),
        'aspect': terrain.select('aspect'),
        'hillshade': terrain.select('hillshade'),
    }

    # Calculate curvature (for identifying ridges/valleys)
    # Curvature = slope of slope
    slope = terrain.select('slope')
    xy = ee.Kernel.gaussian(radius=3, sigma=1, units='pixels', normalize=True)
    curvature = slope.convolve(xy).rename('curvature')
    indices['curvature'] = curvature

    print("  ✓ Slope, aspect, hillshade, curvature")

    return indices


def detect_lineaments_multidirectional(dem, region):
    """
    Detect lineaments (potential faults) using multi-directional edge detection.

    This is critical for Willow deposits which are structurally controlled.

    Args:
        dem: ee.Image - Digital elevation model
        region: ee.Geometry - Study area

    Returns:
        ee.Image - Lineament density (normalized 0-1)
    """
    print("\nDetecting lineaments (potential faults)...")

    # Define directional kernels for edge detection
    # We focus on N-S and NE-SW trends (characteristic of Willow)
    kernels = {
        'N-S': ee.Kernel.fixed(3, 3, [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]),
        'NE-SW': ee.Kernel.fixed(3, 3, [
            [0, 1, 2],
            [-1, 0, 1],
            [-2, -1, 0]
        ]),
        'E-W': ee.Kernel.fixed(3, 3, [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]),
        'NW-SE': ee.Kernel.fixed(3, 3, [
            [-2, -1, 0],
            [-1, 0, 1],
            [0, 1, 2]
        ]),
    }

    # Apply each kernel
    lineament_images = []
    for direction, kernel in kernels.items():
        edge = dem.convolve(kernel).abs()
        lineament_images.append(edge)
        print(f"  ✓ {direction} lineaments")

    # Combine all directions (max value)
    lineaments = ee.ImageCollection(lineament_images).max()

    # Normalize
    lineaments = auto_normalize(lineaments.rename('lineaments'), region)

    return lineaments


def calculate_fault_proximity(lineaments, region):
    """
    Calculate proximity to detected faults/lineaments.

    Args:
        lineaments: ee.Image - Lineament detection image
        region: ee.Geometry - Study area

    Returns:
        ee.Image - Fault proximity score (normalized 0-1, higher = closer to faults)
    """
    print("\nCalculating fault proximity...")

    # Threshold lineaments to get high-confidence faults
    threshold = config.STRUCTURAL_PARAMS['lineament_threshold']
    faults = lineaments.gt(threshold)

    # Calculate distance to nearest fault
    distance = faults.fastDistanceTransform().sqrt().multiply(30)  # Convert to meters

    # Invert and normalize (closer = higher score)
    buffer_distance = config.STRUCTURAL_PARAMS['fault_buffer_distance']
    proximity = ee.Image(buffer_distance).subtract(distance).divide(buffer_distance).clamp(0, 1)

    print("  ✓ Fault proximity calculated")

    return proximity.rename('fault_proximity')


# ============================================================================
# FEATURE STACK ASSEMBLY
# ============================================================================

def create_feature_stack(spectral_indices, structural_indices):
    """
    Create a multi-band feature stack for ML classification.

    Args:
        spectral_indices: dict - Spectral indices
        structural_indices: dict - Structural/terrain indices

    Returns:
        ee.Image - Multi-band feature stack
    """
    print("\nCreating feature stack for ML...")

    bands = []
    band_names = []

    # Add spectral indices
    for name, image in spectral_indices.items():
        if name != 'ndvi':  # Skip NDVI (used for masking, not ML)
            bands.append(image)
            band_names.append(name)

    # Add structural indices
    for name, image in structural_indices.items():
        bands.append(image)
        band_names.append(name)

    # Combine into single image
    feature_stack = ee.Image.cat(bands).rename(band_names)

    print(f"  ✓ Feature stack created with {len(band_names)} bands")
    print(f"    Bands: {', '.join(band_names)}")

    return feature_stack


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def get_image_stats(image, region, band_name=None):
    """
    Get statistics for an image.

    Args:
        image: ee.Image
        region: ee.Geometry
        band_name: str - Optional specific band

    Returns:
        dict - Statistics
    """
    if band_name:
        image = image.select(band_name)

    stats = image.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ).combine(
            reducer2=ee.Reducer.minMax(),
            sharedInputs=True
        ),
        geometry=region,
        scale=30,
        maxPixels=1e9,
        bestEffort=True
    ).getInfo()

    return stats


def print_image_info(image, name="Image"):
    """
    Print information about an image.

    Args:
        image: ee.Image
        name: str - Name for display
    """
    print(f"\n{name}:")
    print(f"  Bands: {image.bandNames().getInfo()}")
    print(f"  Projection: {image.projection().crs().getInfo()}")


if __name__ == "__main__":
    print("Utility Functions Module")
    print("=" * 70)
    print("\nAvailable functions:")
    print("  - get_optimized_landsat_composite()")
    print("  - get_dem_data()")
    print("  - normalize_index_percentile()")
    print("  - calculate_willow_spectral_indices()")
    print("  - detect_lineaments_multidirectional()")
    print("  - calculate_fault_proximity()")
    print("  - create_feature_stack()")
    print("=" * 70)
