# Mat-Su Lode Gold Prospectivity System - Optimization Plan

## Executive Summary

**Current State**: Basic Landsat band ratio script with poor discrimination (everything shows as red)

**Target State**: Production-ready lode gold prospectivity tool validated against known successful mines in Mat-Su/Willow area

**Timeline**: 2-3 weeks for production-ready MVP

**Success Metric**: ≥80% of user's successful mine sites detected in "High" prospectivity zones

---

## Phase 0: Assessment & Setup (Days 1-2)

### Current System Analysis

**What We Have:**
- ✓ Basic Python script (`gold_deposit_detection.py`)
- ✓ GEE authentication working (`ee-markjamesp` project)
- ✓ Required packages installed (earthengine-api, geopandas, rasterio, folium)
- ✓ Understanding of target area (Mat-Su/Willow, Alaska)
- ✓ User has validation data (successful mine locations)

**Critical Problems to Fix:**
1. ✗ Poor normalization → everything shows as anomalous
2. ✗ No machine learning → just simple ratios
3. ✗ No structural analysis → missing key deposit controls
4. ✗ No validation → can't measure success
5. ✗ No integration with known deposits → not learning from data
6. ✗ Generic weights → not tuned for Willow-type deposits

### Setup Actions

**Day 1 Morning: Data Acquisition**
```
Priority 1: User's Validation Dataset
├── Get user's mine location file (CSV/Excel/Shapefile)
├── Required columns: Name, Latitude, Longitude
├── Verify coordinate system (should be WGS84)
├── Save to: data/user_validation_mines.csv
└── Expected: 5-20 successful mine sites

Priority 2: USGS MRDS Data
├── Download: https://mrdata.usgs.gov/mrds/
├── Filter: State=Alaska, Commodity=Gold, Type=Lode
├── Region: Willow Creek district (61.7-62.0°N, -149.5 to -149.0°W)
├── Save to: data/usgs_mrds_alaska_lode.geojson
└── Expected: 30-40 lode gold occurrences

Priority 3: Study Area Definition
├── Confirm bounds with user: [-149.50, 61.70, -149.00, 62.00]
├── Or use user's custom area of interest
└── Save to: config.py
```

**Day 1 Afternoon: Environment Verification**
```
✓ Test GEE connection: ee.Initialize(project='ee-markjamesp')
✓ Test data availability in study area:
  - Landsat 8/9 coverage (2023-2024)
  - ALOS DEM availability
  - ASTER availability (check - may be limited)
✓ Create project directory structure
✓ Set up version control (git)
```

**Day 2: Baseline Metrics**
```
Run existing script on Mat-Su area
├── Document current results
├── Calculate what % of known mines it detects
├── Measure discrimination (% area shown as "High")
├── This is our "before" benchmark
└── Expected: <30% detection, >50% area as "High" (poor)
```

---

## Phase 1: Data Pipeline Optimization (Days 3-5)

### Objective: Get high-quality, clean input data

### 1.1 Satellite Image Composite (Day 3)

**Problem**: Current script uses simple median, doesn't handle Alaska-specific issues

**Optimization Strategy:**

```python
# BEFORE (current script)
def get_landsat_composite(region, start_date, end_date, max_cloud_cover):
    collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filterBounds(region)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover)))
    return collection.median().clip(region)

# AFTER (optimized)
def get_optimized_landsat_composite(region):
    """
    Optimized for Alaska: seasonal filtering, cloud/snow masking, quality scoring
    """
    # 1. Temporal optimization: Late summer only (July-September)
    #    Why: Best vegetation expression, minimal snow, maximum bedrock exposure
    collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
                  .filterBounds(region)
                  .filter(ee.Filter.calendarRange(7, 9, 'month'))  # July-Sept
                  .filter(ee.Filter.calendarRange(2020, 2024, 'year')))  # Multi-year

    # 2. Cloud/Snow masking using QA_PIXEL band
    def mask_clouds_snow(image):
        qa = image.select('QA_PIXEL')
        # Bit masks for cloud, cloud shadow, snow
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)  # Cloud
        shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)  # Cloud shadow
        snow_mask = qa.bitwiseAnd(1 << 5).eq(0)    # Snow/ice

        return image.updateMask(cloud_mask.And(shadow_mask).And(snow_mask))

    # 3. Quality scoring: Prefer high sun angle, low sensor zenith
    def add_quality_score(image):
        sun_elevation = image.get('SUN_ELEVATION')
        # Higher sun = better illumination
        quality = ee.Number(sun_elevation).divide(90)
        return image.set('quality', quality)

    # 4. Apply masks and scoring
    collection = collection.map(mask_clouds_snow).map(add_quality_score)

    # 5. Quality mosaic (weighted by quality score, not just median)
    composite = collection.qualityMosaic('quality').clip(region)

    # 6. Apply surface reflectance scaling
    def apply_scale_factors(image):
        optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        return image.addBands(optical, None, True)

    return apply_scale_factors(composite)
```

**Expected Improvement**:
- Better signal-to-noise in mineral indices
- Less contamination from clouds/snow
- More consistent across study area

**Validation**: Visual inspection - should see clear bedrock in alpine areas

### 1.2 DEM Selection & Preprocessing (Day 3)

**Problem**: SRTM has voids in Alaska

**Optimization:**

```python
# Use ALOS World 3D instead of SRTM
def get_optimized_dem(region):
    """
    ALOS is superior to SRTM for high latitudes.
    Apply void filling and smoothing.
    """
    # Primary: ALOS
    alos_dem = ee.Image('JAXA/ALOS/AW3D30/V3_2').select('AVE_DSM')

    # Fill any remaining voids with SRTM as backup
    srtm_dem = ee.Image('USGS/SRTMGL1_003').select('elevation')

    # Mosaic: ALOS preferred, SRTM fills gaps
    dem = ee.ImageCollection([alos_dem, srtm_dem]).mosaic().clip(region)

    # Smoothing: Reduce noise for derivative calculations
    dem_smooth = dem.focalMedian(radius=1, units='pixels')  # 3x3 median

    return dem_smooth
```

**Expected Improvement**:
- No voids in study area
- Cleaner derivatives (slope, curvature)
- Better lineament detection

### 1.3 ASTER Data Integration (Day 4)

**Problem**: Landsat alone may not capture all alteration types

**Strategy**: Add ASTER if available, graceful fallback if not

```python
def get_aster_data_if_available(region):
    """
    ASTER has superior SWIR bands for alteration, but coverage may be spotty.
    """
    aster_collection = (ee.ImageCollection('ASTER/AST_L1T_003')
                        .filterBounds(region)
                        .filter(ee.Filter.calendarRange(2000, 2024, 'year')))

    # Check if any images available
    count = aster_collection.size().getInfo()

    if count > 0:
        print(f"✓ ASTER data available: {count} scenes")
        # Use best quality scene or mosaic
        aster_composite = aster_collection.qualityMosaic('VNIR_Band3N')
        return aster_composite, True
    else:
        print("⚠ No ASTER data - will use Landsat-only indices")
        return None, False
```

**Decision Point**:
- If ASTER available → Use advanced alteration indices
- If not → Optimize Landsat-based proxies

**Expected Improvement**: Better discrimination of alteration types (if available)

### 1.4 Training Data Preparation (Day 4-5)

**Critical**: This is where the model learns what Willow-type deposits look like

**Optimization:**

```python
def prepare_training_dataset(user_mines_file, usgs_mrds_file, region):
    """
    Combine user's high-confidence mines with USGS data.
    Create balanced positive/negative samples.
    """
    import geopandas as gpd
    import pandas as pd

    # 1. Load user's validation mines (HIGH CONFIDENCE)
    user_mines = pd.read_csv(user_mines_file)
    user_gdf = gpd.GeoDataFrame(
        user_mines,
        geometry=gpd.points_from_xy(user_mines.Longitude, user_mines.Latitude),
        crs='EPSG:4326'
    )
    user_gdf['confidence'] = 'HIGH'
    user_gdf['source'] = 'USER'

    # 2. Load USGS MRDS (MEDIUM CONFIDENCE)
    usgs_gdf = gpd.read_file(usgs_mrds_file)
    usgs_gdf['confidence'] = 'MEDIUM'
    usgs_gdf['source'] = 'USGS'

    # 3. Filter USGS to study area
    usgs_filtered = usgs_gdf.cx[region[0]:region[2], region[1]:region[3]]

    # 4. Remove duplicates (user mines may overlap with USGS)
    #    Buffer 500m - if USGS point within 500m of user point, remove USGS
    user_buffer = user_gdf.buffer(0.005)  # ~500m at this latitude
    usgs_unique = usgs_filtered[~usgs_filtered.within(user_buffer.unary_union)]

    # 5. Combine
    positive_samples = pd.concat([
        user_gdf[['geometry', 'confidence', 'source']],
        usgs_unique[['geometry', 'confidence', 'source']]
    ])

    print(f"Positive samples: {len(positive_samples)}")
    print(f"  - User mines (high confidence): {len(user_gdf)}")
    print(f"  - USGS mines (medium confidence): {len(usgs_unique)}")

    # 6. Generate negative samples (2x positive samples)
    num_negative = len(positive_samples) * 2

    # Exclusion buffer: 5km from any known deposit
    exclusion_buffer = positive_samples.buffer(0.05).unary_union  # ~5km

    # Sample area = region - exclusion buffer
    from shapely.geometry import box
    region_poly = box(region[0], region[1], region[2], region[3])
    sample_area = region_poly.difference(exclusion_buffer)

    # Generate random points
    negative_points = []
    while len(negative_points) < num_negative:
        # Random point in bounding box
        lon = random.uniform(region[0], region[2])
        lat = random.uniform(region[1], region[3])
        point = Point(lon, lat)

        # Check if in sample area
        if sample_area.contains(point):
            negative_points.append(point)

    negative_gdf = gpd.GeoDataFrame(
        {'geometry': negative_points, 'class': 0},
        crs='EPSG:4326'
    )

    positive_samples['class'] = 1

    print(f"Negative samples: {len(negative_gdf)}")
    print(f"Total training points: {len(positive_samples) + len(negative_gdf)}")

    return positive_samples, negative_gdf
```

**Expected Improvement**:
- High-confidence user data gets priority
- Balanced dataset prevents bias
- Spatial separation ensures independence

---

## Phase 2: Feature Engineering Optimization (Days 6-8)

### Objective: Extract maximum information from data

### 2.1 Spectral Indices - Willow-Specific (Day 6)

**Problem**: Generic indices don't capture Willow deposit characteristics

**Optimization: Calibrate for orogenic gold**

```python
def calculate_willow_specific_indices(image):
    """
    Indices tuned for orogenic lode gold (Willow-type).
    Emphasize carbonate + sericite + gossan signature.
    """
    indices = {}

    # 1. Iron Oxide (Gossan) - CRITICAL
    # Optimize for weathered pyrite/arsenopyrite
    indices['iron_oxide'] = image.select('SR_B4').divide(image.select('SR_B2'))

    # 2. Carbonate Alteration - DIAGNOSTIC for Willow
    # Ankerite/calcite have distinct SWIR signature
    # Ratio emphasizes carbonate absorption
    indices['carbonate'] = image.select('SR_B7').divide(image.select('SR_B6'))

    # 3. Sericite (Phyllic alteration)
    # White mica associated with gold veins
    indices['sericite'] = (image.select('SR_B5').multiply(image.select('SR_B7'))
                           .divide(image.select('SR_B6').pow(2)))

    # 4. Chlorite (Distal alteration)
    # Fe-Mg chlorite in propylitic zones
    indices['chlorite'] = image.select('SR_B6').divide(image.select('SR_B5'))

    # 5. Composite Alteration - Willow Signature
    # Weight carbonate heavily (diagnostic)
    indices['willow_signature'] = (
        indices['iron_oxide'].multiply(0.30).add(
        indices['carbonate'].multiply(0.40)).add(    # Carbonate weighted highest
        indices['sericite'].multiply(0.20)).add(
        indices['chlorite'].multiply(0.10))
    )

    # 6. NDVI for vegetation masking
    indices['ndvi'] = (image.select('SR_B5').subtract(image.select('SR_B4'))
                       .divide(image.select('SR_B5').add(image.select('SR_B4'))))

    return indices
```

**Key Innovation**: Carbonate signature weighted 40% (diagnostic for Willow)

### 2.2 Proper Normalization (Day 6) - CRITICAL FIX

**Problem**: Current script doesn't normalize properly → everything red

**Solution**: Percentile-based normalization per index

```python
def normalize_index_robust(index_image, region, percentile_low=5, percentile_high=95):
    """
    Robust normalization using percentiles.
    This fixes the "everything is red" problem.
    """
    # Calculate percentiles over study area
    stats = index_image.reduceRegion(
        reducer=ee.Reducer.percentile([percentile_low, percentile_high]),
        geometry=region,
        scale=30,
        maxPixels=1e9,
        bestEffort=True
    )

    # Get percentile values
    band_name = index_image.bandNames().get(0)
    p_low = ee.Number(stats.get(f'{band_name}_p{percentile_low}'))
    p_high = ee.Number(stats.get(f'{band_name}_p{percentile_high}'))

    # Normalize to 0-1 range
    normalized = (index_image.subtract(p_low)
                  .divide(p_high.subtract(p_low))
                  .clamp(0, 1))

    # Debug output
    print(f"  {band_name}: p{percentile_low}={p_low.getInfo():.3f}, p{percentile_high}={p_high.getInfo():.3f}")

    return normalized

# Apply to all indices
def normalize_all_indices(indices, region):
    """
    Normalize each index independently.
    """
    normalized = {}
    print("Normalizing indices:")
    for key, index in indices.items():
        if key != 'ndvi':  # Don't normalize NDVI (already -1 to 1)
            normalized[key] = normalize_index_robust(index, region)
    normalized['ndvi'] = indices['ndvi']

    return normalized
```

**Expected Improvement**:
- Only top 5-10% of values become "high"
- Clear distinction between anomalies and background
- **This is the single most important fix**

### 2.3 Structural Features - Fault Detection (Day 7)

**Problem**: Willow deposits are structurally controlled - current script ignores this

**Optimization: Multi-azimuth lineament detection**

```python
def detect_structural_features_optimized(dem, region):
    """
    Detect faults/shears using multiple techniques.
    Focus on N-NNE trend (dominant in Willow).
    """
    features = {}

    # Basic derivatives
    features['elevation'] = dem
    features['slope'] = ee.Terrain.slope(dem)
    features['aspect'] = ee.Terrain.aspect(dem)

    # Lineament detection: Multi-azimuth edge enhancement
    azimuths = [0, 30, 60, 90, 120, 150]  # Cover all orientations
    lineament_layers = []

    for azimuth in azimuths:
        # Hillshade at this azimuth
        hillshade = ee.Terrain.hillshade(dem, azimuth=azimuth, elevation=45)

        # Edge detection (Sobel filter)
        edges = hillshade.convolve(ee.Kernel.sobel())

        lineament_layers.append(edges)

    # Combine: Maximum response across all azimuths
    lineament_composite = ee.ImageCollection(lineament_layers).max()

    # Threshold to get significant lineaments
    lineament_threshold = lineament_composite.gt(threshold_value)

    # Lineament density: Focal sum in 500m radius
    lineament_density = lineament_threshold.focalSum(
        radius=500,
        units='meters',
        kernelType='circle'
    )

    features['lineament_density'] = lineament_density

    # Distance to nearest lineament
    distance_to_lineament = (lineament_threshold
                             .fastDistanceTransform()
                             .sqrt()
                             .multiply(30))  # Convert to meters

    features['distance_to_lineament'] = distance_to_lineament

    # Fault intersections: Areas with high lineament density
    # Indicates multiple structures crossing
    intersections = lineament_density.gt(3)  # At least 3 lineaments nearby
    features['fault_intersections'] = intersections

    # N-NNE lineament density (Willow-specific)
    # Emphasis on north-trending structures
    n_lineaments = []
    for az in [0, 15, 30, 345]:  # North ± 30°
        hs = ee.Terrain.hillshade(dem, azimuth=az, elevation=45)
        edge = hs.convolve(ee.Kernel.sobel())
        n_lineaments.append(edge)

    n_lineament_density = ee.ImageCollection(n_lineaments).max().focalSum(500, 'meters')
    features['n_lineament_density'] = n_lineament_density

    # Topographic Position Index (ridges vs valleys)
    # Veins often exposed on ridges
    tpi = dem.subtract(dem.focalMean(radius=500, units='meters'))
    features['tpi'] = tpi

    # Terrain Ruggedness (fault zones are rough)
    tri = (dem.subtract(dem.focalMedian(radius=30, units='meters'))
           .abs()
           .focalSum(radius=90, units='meters'))
    features['tri'] = tri

    return features
```

**Expected Improvement**:
- Captures fault-controlled deposit locations
- N-trending emphasis matches Willow geology
- Fault intersections are prime targets

### 2.4 Feature Stack Assembly (Day 8)

**Combine all features into one image for ML**

```python
def create_feature_stack(spectral_indices, structural_features, region):
    """
    Combine all features into single multi-band image.
    Normalize everything to 0-1 for ML.
    """
    # Normalize spectral indices (already done)
    # Normalize structural features

    struct_normalized = {}
    for key, feature in structural_features.items():
        if key not in ['elevation', 'slope', 'aspect']:  # These need special handling
            struct_normalized[key] = normalize_index_robust(feature, region)

    # Elevation: Normalize to study area range
    elev_stats = structural_features['elevation'].reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=region,
        scale=30,
        maxPixels=1e9
    )
    elev_min = ee.Number(elev_stats.get('elevation_min'))
    elev_max = ee.Number(elev_stats.get('elevation_max'))
    struct_normalized['elevation'] = (structural_features['elevation']
                                       .subtract(elev_min)
                                       .divide(elev_max.subtract(elev_min)))

    # Slope: Normalize to 0-90°
    struct_normalized['slope'] = structural_features['slope'].divide(90)

    # Aspect: Convert to northness/eastness
    aspect_rad = structural_features['aspect'].multiply(math.pi/180)
    struct_normalized['northness'] = aspect_rad.cos().add(1).divide(2)  # 0-1
    struct_normalized['eastness'] = aspect_rad.sin().add(1).divide(2)   # 0-1

    # Combine into single image
    feature_bands = []
    band_names = []

    # Add spectral
    for key in ['iron_oxide', 'carbonate', 'sericite', 'chlorite', 'willow_signature']:
        feature_bands.append(spectral_indices[key])
        band_names.append(key)

    # Add structural
    for key in ['lineament_density', 'distance_to_lineament', 'n_lineament_density',
                'tpi', 'tri', 'elevation', 'slope', 'northness', 'eastness']:
        feature_bands.append(struct_normalized[key])
        band_names.append(key)

    # Stack into one image
    feature_stack = ee.Image.cat(feature_bands).rename(band_names)

    print(f"Feature stack created: {len(band_names)} bands")
    print(f"Bands: {', '.join(band_names)}")

    return feature_stack, band_names
```

**Result**: Single image with ~14 normalized features ready for ML

---

## Phase 3: Machine Learning Optimization (Days 9-11)

### Objective: Build classifier that learns Willow deposit signature

### 3.1 Feature Extraction & Dataset Creation (Day 9)

```python
def extract_training_features(feature_stack, positive_samples, negative_samples):
    """
    Extract feature values at training point locations.
    """
    # Convert geopandas to Earth Engine FeatureCollections
    def gdf_to_ee_fc(gdf, class_label):
        features = []
        for idx, row in gdf.iterrows():
            point = ee.Geometry.Point([row.geometry.x, row.geometry.y])
            props = {'class': class_label}
            if 'confidence' in row:
                props['confidence'] = row['confidence']
            feature = ee.Feature(point, props)
            features.append(feature)
        return ee.FeatureCollection(features)

    positive_fc = gdf_to_ee_fc(positive_samples, 1)
    negative_fc = gdf_to_ee_fc(negative_samples, 0)

    # Sample feature stack at points
    positive_features = feature_stack.sampleRegions(
        collection=positive_fc,
        properties=['class', 'confidence'],
        scale=30,
        geometries=False
    )

    negative_features = feature_stack.sampleRegions(
        collection=negative_fc,
        properties=['class'],
        scale=30,
        geometries=False
    )

    # Combine
    training_data = positive_features.merge(negative_features)

    # Split train/test (70/30)
    training_data = training_data.randomColumn('random', seed=42)
    train = training_data.filter(ee.Filter.lt('random', 0.7))
    test = training_data.filter(ee.Filter.gte('random', 0.7))

    print(f"Training samples: {train.size().getInfo()}")
    print(f"Test samples: {test.size().getInfo()}")

    return train, test, training_data
```

### 3.2 Random Forest Training with Optimization (Day 9-10)

```python
def train_optimized_random_forest(training_data, feature_names):
    """
    Train Random Forest with hyperparameter optimization.
    """
    # Hyperparameters to test
    param_grid = {
        'numberOfTrees': [100, 300, 500],
        'variablesPerSplit': [None, 3, 5],
        'minLeafPopulation': [1, 5, 10]
    }

    best_accuracy = 0
    best_classifier = None
    best_params = None

    print("Hyperparameter search:")

    for n_trees in param_grid['numberOfTrees']:
        for vars_per_split in param_grid['variablesPerSplit']:
            for min_leaf in param_grid['minLeafPopulation']:

                # Train classifier
                classifier = ee.Classifier.smileRandomForest(
                    numberOfTrees=n_trees,
                    variablesPerSplit=vars_per_split,
                    minLeafPopulation=min_leaf,
                    seed=42
                ).train(
                    features=training_data,
                    classProperty='class',
                    inputProperties=feature_names
                )

                # Validate on test set
                test_classified = test.classify(classifier)
                confusion = test_classified.errorMatrix('class', 'classification')
                accuracy = confusion.accuracy().getInfo()

                print(f"  Trees={n_trees}, Vars={vars_per_split}, MinLeaf={min_leaf}: Acc={accuracy:.3f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_classifier = classifier
                    best_params = {'trees': n_trees, 'vars': vars_per_split, 'min_leaf': min_leaf}

    print(f"\nBest model: {best_params}, Accuracy: {best_accuracy:.3f}")

    # Feature importance
    importance = best_classifier.explain().getInfo()
    print("\nFeature Importance (top 5):")
    # Extract and sort importance
    # (implementation depends on GEE classifier output format)

    return best_classifier, best_params, best_accuracy
```

### 3.3 Validation Loop (Day 10-11) - CRITICAL

**This is where we iterate until the model works**

```python
def validate_model_against_user_mines(classifier, feature_stack, user_mines_gdf):
    """
    Test model on user's known successful mines.
    This is the ultimate validation.
    """
    # Classify entire study area
    classified = feature_stack.classify(classifier)
    probability = feature_stack.classify(classifier.setOutputMode('PROBABILITY'))

    results = []

    for idx, mine in user_mines_gdf.iterrows():
        mine_name = mine.get('Name', f'Mine {idx+1}')
        lon, lat = mine.geometry.x, mine.geometry.y

        # Sample probability at mine location
        point = ee.Geometry.Point([lon, lat])
        sample = probability.sampleRegions(
            collection=ee.FeatureCollection([ee.Feature(point)]),
            scale=30
        ).first()

        prob = sample.get('classification').getInfo()

        # Classify
        if prob >= 0.7:
            classification = 'HIGH'
        elif prob >= 0.5:
            classification = 'MEDIUM'
        else:
            classification = 'LOW'

        results.append({
            'mine': mine_name,
            'probability': prob,
            'classification': classification,
            'lat': lat,
            'lon': lon
        })

    # Calculate success rate
    high_detections = sum(1 for r in results if r['classification'] == 'HIGH')
    medium_or_high = sum(1 for r in results if r['classification'] in ['HIGH', 'MEDIUM'])

    success_rate_high = high_detections / len(results)
    success_rate_med_high = medium_or_high / len(results)

    # Display results
    print("\n" + "="*70)
    print("VALIDATION AGAINST USER'S SUCCESSFUL MINES")
    print("="*70)

    for r in results:
        symbol = "✓" if r['classification'] == 'HIGH' else "⚠" if r['classification'] == 'MEDIUM' else "✗"
        print(f"{symbol} {r['mine']:30s} Prob: {r['probability']:.3f} ({r['classification']})")

    print(f"\nSuccess Rate (High): {success_rate_high:.1%} ({high_detections}/{len(results)})")
    print(f"Success Rate (Medium+): {success_rate_med_high:.1%} ({medium_or_high}/{len(results)})")

    # Pass/Fail
    if success_rate_high >= 0.8:
        print("\n✓ VALIDATION PASSED - Model is ready for production")
        return True, results
    else:
        print("\n✗ VALIDATION FAILED - Model needs tuning")
        print("\nDiagnostic steps:")
        print("1. Check which mines were missed")
        print("2. Extract feature values at those locations")
        print("3. Identify which features are weak")
        print("4. Adjust feature weights or add new features")
        return False, results
```

**Iteration Strategy if Validation Fails:**

```python
def debug_failed_detections(results, feature_stack, user_mines_gdf):
    """
    Analyze why certain mines weren't detected.
    """
    failed_mines = [r for r in results if r['classification'] != 'HIGH']

    print("\nAnalyzing failed detections:")

    for mine in failed_mines:
        print(f"\n{mine['mine']}:")

        # Extract all feature values at this location
        point = ee.Geometry.Point([mine['lon'], mine['lat']])
        features = feature_stack.sampleRegions(
            collection=ee.FeatureCollection([ee.Feature(point)]),
            scale=30
        ).first().getInfo()['properties']

        # Identify weak features
        print("  Feature values:")
        for key, value in features.items():
            if value < 0.3:  # Low value
                print(f"    ✗ {key}: {value:.3f} (LOW)")
            elif value > 0.7:  # High value
                print(f"    ✓ {key}: {value:.3f} (HIGH)")
            else:
                print(f"    - {key}: {value:.3f}")

        # Recommendation
        print("  Recommendation:")
        # Logic to suggest fixes based on which features are low
```

**Expected Iteration**:
- First run: 60-70% success rate (typical)
- Analyze failures, adjust weights
- Second run: 75-85% success rate
- Third run (if needed): >80% success rate

---

## Phase 4: Prospectivity Model Integration (Days 12-13)

### Objective: Combine ML with weighted evidence layers

### 4.1 Multi-Criteria Decision Analysis (Day 12)

```python
def create_optimized_prospectivity_model(
    spectral_indices,
    structural_features,
    ml_probability,
    region
):
    """
    Weighted overlay calibrated to Willow deposits.
    Weights adjusted based on validation results.
    """
    # Normalize if not already
    spectral_norm = spectral_indices['willow_signature']  # Already normalized

    # Structural composite
    structural_composite = (
        structural_features['lineament_density'].multiply(0.4).add(
        structural_features['n_lineament_density'].multiply(0.3)).add(
        structural_features['fault_intersections'].multiply(0.3))
    )
    structural_norm = normalize_index_robust(structural_composite, region)

    # ML probability (already 0-1)
    ml_norm = ml_probability

    # Weighted combination
    # Weights based on Willow deposit controls:
    # - Spectral (alteration): 30%
    # - Structural (faults): 45% (CRITICAL for orogenic gold)
    # - ML (integrated learning): 25%

    prospectivity = (
        spectral_norm.multiply(0.30).add(
        structural_norm.multiply(0.45)).add(
        ml_norm.multiply(0.25))
    )

    # Apply masks
    # 1. Vegetation mask
    veg_mask = spectral_indices['ndvi'].lt(0.4)

    # 2. Slope mask (focus on bedrock exposures)
    slope_mask = structural_features['slope'].gt(15)  # >15° likely bedrock

    # Combined mask
    combined_mask = veg_mask.And(slope_mask)

    prospectivity_masked = prospectivity.updateMask(combined_mask)

    return prospectivity_masked
```

### 4.2 Threshold Calibration (Day 12)

```python
def calibrate_thresholds(prospectivity_image, region, known_deposits_fc):
    """
    Calibrate High/Medium/Low thresholds based on known deposits.
    Goal: 80% of known deposits in "High" zone.
    """
    # Sample prospectivity at known deposits
    deposit_values = prospectivity_image.sampleRegions(
        collection=known_deposits_fc,
        scale=30,
        geometries=False
    )

    # Get values as list
    values = deposit_values.aggregate_array('prospectivity').getInfo()
    values_sorted = sorted(values, reverse=True)

    # 80th percentile of known deposits = threshold for "High"
    # This ensures 80% of deposits fall in "High" zone
    threshold_high = values_sorted[int(len(values_sorted) * 0.2)]  # 80th percentile

    # Area percentile: What % of total area is above this threshold?
    area_stats = prospectivity_image.reduceRegion(
        reducer=ee.Reducer.percentile([80, 90, 95, 99]),
        geometry=region,
        scale=30,
        maxPixels=1e9
    )

    p95 = area_stats.get('prospectivity_p95').getInfo()
    p99 = area_stats.get('prospectivity_p99').getInfo()

    print(f"Known deposit 80th percentile: {threshold_high:.3f}")
    print(f"Study area 95th percentile: {p95:.3f}")
    print(f"Study area 99th percentile: {p99:.3f}")

    # Use the more conservative threshold
    final_threshold_high = max(threshold_high, p95)
    threshold_medium = final_threshold_high * 0.7  # 70% of high threshold

    print(f"\nCalibrated Thresholds:")
    print(f"  High: ≥{final_threshold_high:.3f}")
    print(f"  Medium: {threshold_medium:.3f} - {final_threshold_high:.3f}")
    print(f"  Low: <{threshold_medium:.3f}")

    # Classify
    prospectivity_classified = (
        prospectivity_image
        .where(prospectivity_image.lt(threshold_medium), 1)         # Low
        .where(prospectivity_image.gte(threshold_medium).And(
               prospectivity_image.lt(final_threshold_high)), 2)    # Medium
        .where(prospectivity_image.gte(final_threshold_high), 3)    # High
    )

    # Check what % of area is High
    area_fraction = prospectivity_image.gte(final_threshold_high).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=30,
        maxPixels=1e9
    ).get('prospectivity').getInfo()

    print(f"\nArea classified as High: {area_fraction*100:.1f}%")

    if area_fraction > 0.10:
        print("⚠ Warning: >10% of area is High - thresholds may need adjustment")

    return prospectivity_classified, final_threshold_high, threshold_medium
```

---

## Phase 5: Output Optimization (Days 14-15)

### Objective: Production-quality deliverables

### 5.1 Interactive Map Enhancement (Day 14)

```python
def create_production_map(
    region_bounds,
    landsat_composite,
    spectral_indices,
    structural_features,
    prospectivity,
    known_deposits_gdf,
    user_mines_gdf
):
    """
    Professional-grade interactive map.
    """
    import folium
    from folium import plugins

    # Center on study area
    center_lat = (region_bounds[1] + region_bounds[3]) / 2
    center_lon = (region_bounds[0] + region_bounds[2]) / 2

    # Create map with terrain basemap
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap',
        control_scale=True
    )

    # Add Google Satellite basemap option
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # Helper function for GEE layers
    def add_ee_layer(image, vis_params, name, shown=True):
        map_id_dict = ee.Image(image).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name=name,
            overlay=True,
            control=True,
            show=shown
        ).add_to(m)

    # Layer 1: Landsat True Color (default visible)
    add_ee_layer(
        landsat_composite,
        {'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min': 0, 'max': 0.3, 'gamma': 1.4},
        'Landsat True Color',
        shown=True
    )

    # Layer 2: Iron Oxide
    add_ee_layer(
        spectral_indices['iron_oxide'],
        {'min': 0, 'max': 1, 'palette': ['blue', 'cyan', 'yellow', 'orange', 'red']},
        'Iron Oxide (Gossan)',
        shown=False
    )

    # Layer 3: Carbonate Alteration
    add_ee_layer(
        spectral_indices['carbonate'],
        {'min': 0, 'max': 1, 'palette': ['blue', 'green', 'yellow', 'red']},
        'Carbonate Alteration',
        shown=False
    )

    # Layer 4: Lineament Density
    add_ee_layer(
        structural_features['lineament_density'],
        {'min': 0, 'max': 1, 'palette': ['white', 'yellow', 'orange', 'red']},
        'Lineament Density',
        shown=False
    )

    # Layer 5: Prospectivity (default visible)
    add_ee_layer(
        prospectivity,
        {'min': 0, 'max': 1, 'palette': ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']},
        'Gold Prospectivity',
        shown=True
    )

    # Add known deposits as markers
    # USGS MRDS deposits (yellow circles)
    for idx, deposit in known_deposits_gdf.iterrows():
        folium.CircleMarker(
            location=[deposit.geometry.y, deposit.geometry.x],
            radius=5,
            popup=f"<b>{deposit.get('site_name', 'Unknown')}</b><br>Type: {deposit.get('dep_type', 'N/A')}",
            color='yellow',
            fill=True,
            fillColor='yellow',
            fillOpacity=0.6
        ).add_to(m)

    # User's successful mines (red stars - highest priority)
    for idx, mine in user_mines_gdf.iterrows():
        folium.Marker(
            location=[mine.geometry.y, mine.geometry.x],
            popup=f"<b>★ {mine.get('Name', f'Mine {idx+1}')}</b><br>Successful Producer",
            icon=folium.Icon(color='red', icon='star', prefix='fa')
        ).add_to(m)

    # Add measurement tool
    plugins.MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m)

    # Add fullscreen option
    plugins.Fullscreen(position='topright').add_to(m)

    # Add layer control
    folium.LayerControl(position='topleft', collapsed=False).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 220px; height: 180px;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin:0; font-weight:bold;">Gold Prospectivity</p>
    <p style="margin:2px;"><span style="color:red;">●</span> High (>0.7)</p>
    <p style="margin:2px;"><span style="color:orange;">●</span> Medium (0.5-0.7)</p>
    <p style="margin:2px;"><span style="color:green;">●</span> Low (<0.5)</p>
    <br>
    <p style="margin:0; font-weight:bold;">Known Deposits</p>
    <p style="margin:2px;"><span style="color:red;">★</span> User Mines (validated)</p>
    <p style="margin:2px;"><span style="color:yellow;">●</span> USGS MRDS</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m
```

### 5.2 Target Extraction & Ranking (Day 14)

```python
def extract_top_targets_optimized(
    prospectivity_image,
    feature_stack,
    region,
    num_targets=100,
    min_threshold=0.7
):
    """
    Extract and rank high-priority targets.
    """
    # Threshold for high prospectivity
    high_zones = prospectivity_image.gt(min_threshold)

    # Convert to vectors (polygons)
    vectors = high_zones.reduceToVectors(
        geometry=region,
        scale=30,
        geometryType='centroid',
        maxPixels=1e9,
        bestEffort=True
    )

    # Sample all feature values at each anomaly
    targets = feature_stack.addBands(prospectivity_image.rename('prospectivity')).sampleRegions(
        collection=vectors,
        scale=30,
        geometries=True
    )

    # Sort by prospectivity
    targets_sorted = targets.sort('prospectivity', False).limit(num_targets)

    # Convert to list
    target_list = targets_sorted.getInfo()['features']

    # Format as DataFrame
    import pandas as pd
    data = []

    for i, target in enumerate(target_list, 1):
        coords = target['geometry']['coordinates']
        props = target['properties']

        data.append({
            'Rank': i,
            'Longitude': coords[0],
            'Latitude': coords[1],
            'Prospectivity': props['prospectivity'],
            'Iron_Oxide': props.get('iron_oxide', 0),
            'Carbonate': props.get('carbonate', 0),
            'Sericite': props.get('sericite', 0),
            'Lineament_Density': props.get('lineament_density', 0),
            'N_Lineament_Density': props.get('n_lineament_density', 0),
            'Elevation_m': props.get('elevation', 0) * 1000,  # Denormalize
            'Slope_deg': props.get('slope', 0) * 90,
            'Distance_to_Known_m': calculate_distance_to_nearest_mine(coords, known_mines)
        })

    df = pd.DataFrame(data)

    # Add interpretation column
    def interpret_target(row):
        notes = []
        if row['Iron_Oxide'] > 0.8:
            notes.append("Strong gossan")
        if row['Carbonate'] > 0.8:
            notes.append("Carbonate alteration")
        if row['N_Lineament_Density'] > 0.7:
            notes.append("N-trending structure")
        if row['Distance_to_Known_m'] < 2000:
            notes.append(f"Near known deposit ({row['Distance_to_Known_m']:.0f}m)")

        return "; ".join(notes) if notes else "New prospect area"

    df['Notes'] = df.apply(interpret_target, axis=1)

    return df
```

### 5.3 Validation Report Generation (Day 15)

```python
def generate_validation_report(
    validation_results,
    model_metrics,
    feature_importance,
    output_file='outputs/validation/validation_report.md'
):
    """
    Professional validation report in Markdown format.
    """
    report = f"""
# Mat-Su Lode Gold Prospectivity Model - Validation Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Study Area**: Willow Creek Mining District, Alaska
**Model Version**: 1.0

---

## Executive Summary

This model was developed to identify prospective lode gold targets in the Mat-Su Valley using satellite remote sensing and machine learning.

**Key Results**:
- **Validation Success Rate**: {validation_results['success_rate_high']:.1%} (High zone)
- **Model Accuracy**: {model_metrics['accuracy']:.1%}
- **High Prospectivity Area**: {validation_results['area_fraction']*100:.1f}% of study area
- **Top Targets Identified**: {validation_results['num_targets']}

### Validation Status
"""

    if validation_results['success_rate_high'] >= 0.8:
        report += "✅ **PASSED** - Model successfully detects known lode gold deposits\n\n"
    else:
        report += "⚠️ **NEEDS IMPROVEMENT** - Model requires additional tuning\n\n"

    report += f"""
---

## Methodology

### Data Sources
1. **Satellite Imagery**: Landsat 8/9 (2020-2024, summer months)
2. **Elevation Data**: ALOS World 3D DEM (30m resolution)
3. **Training Data**:
   - User-provided successful mines: {validation_results['num_user_mines']}
   - USGS MRDS lode gold occurrences: {validation_results['num_usgs_mines']}

### Features Calculated
- **Spectral**: Iron oxide, carbonate alteration, sericite, chlorite
- **Structural**: Lineament density, fault intersections, topographic position
- **Machine Learning**: Random Forest classifier ({model_metrics['n_trees']} trees)

### Model Approach
Weighted multi-criteria decision analysis combining:
- Spectral alteration indices (30%)
- Structural controls (45%)
- Machine learning probability (25%)

---

## Validation Results

### Known Mine Detection

| Mine Name | Prospectivity | Classification | Status |
|-----------|---------------|----------------|--------|
"""

    for mine in validation_results['mine_results']:
        symbol = "✅" if mine['classification'] == 'HIGH' else "⚠️" if mine['classification'] == 'MEDIUM' else "❌"
        report += f"| {mine['name']:30s} | {mine['probability']:.3f} | {mine['classification']:10s} | {symbol} |\n"

    report += f"""

**Summary**:
- Detected in High zone: {validation_results['high_count']}/{validation_results['total_mines']} ({validation_results['success_rate_high']:.1%})
- Detected in Medium+ zone: {validation_results['med_high_count']}/{validation_results['total_mines']} ({validation_results['success_rate_med_high']:.1%})

### Model Performance

**Confusion Matrix**:
```
{validation_results['confusion_matrix_str']}
```

**Metrics**:
- Overall Accuracy: {model_metrics['accuracy']:.1%}
- Precision (High class): {model_metrics['precision']:.1%}
- Recall (High class): {model_metrics['recall']:.1%}
- F1 Score: {model_metrics['f1']:.3f}
- Kappa: {model_metrics['kappa']:.3f}

### Feature Importance

Top 5 features contributing to classification:

"""

    for i, (feature, importance) in enumerate(feature_importance[:5], 1):
        report += f"{i}. **{feature}**: {importance:.1%}\n"

    report += f"""

---

## Prospectivity Map Analysis

### Area Distribution

| Class | Threshold | Area (%) |
|-------|-----------|----------|
| High | ≥{validation_results['threshold_high']:.2f} | {validation_results['area_high']:.1f}% |
| Medium | {validation_results['threshold_medium']:.2f}-{validation_results['threshold_high']:.2f} | {validation_results['area_medium']:.1f}% |
| Low | <{validation_results['threshold_medium']:.2f} | {validation_results['area_low']:.1f}% |

### Top 10 Targets

| Rank | Lat | Lon | Prospectivity | Notes |
|------|-----|-----|---------------|-------|
"""

    for target in validation_results['top_10_targets']:
        report += f"| {target['rank']} | {target['lat']:.4f} | {target['lon']:.4f} | {target['score']:.3f} | {target['notes']} |\n"

    report += """

---

## Recommendations

### Field Work Priorities

1. **Immediate Priority** (Rank 1-20)
   - High prospectivity (>0.8)
   - Strong alteration signatures
   - Multiple structural controls

2. **Secondary Priority** (Rank 21-50)
   - Medium-high prospectivity (0.7-0.8)
   - Single dominant feature (e.g., strong gossan OR fault)

3. **Reconnaissance** (Rank 51-100)
   - Medium prospectivity
   - Exploration potential in underexplored areas

### Ground Verification Steps

For each target:
1. Access and permits (land ownership check)
2. Field reconnaissance (geology, structures, alteration)
3. Rock chip sampling (assay for Au, As, Sb pathfinders)
4. Geophysical survey (magnetics, IP if warranted)
5. Detailed mapping and trenching (if promising)

### Model Limitations

- Vegetation obscures bedrock in valleys - focus on alpine exposures
- Glacial cover limits detection at high elevations
- Transported gossan (from placer) may cause false positives
- Model trained on known deposits - may miss novel deposit types

---

## Conclusions

"""

    if validation_results['success_rate_high'] >= 0.8:
        report += """
The model **successfully validates** against known lode gold deposits in the Willow Creek district. The prospectivity map can be used with confidence for exploration targeting.

**Key Strengths**:
- High detection rate of known producers
- Good discrimination (limited high prospectivity area)
- Integration of multiple data sources
- Calibrated to Willow-type orogenic gold deposits

**Next Steps**:
- Apply to adjacent unexplored areas
- Integrate with ground-based data as it becomes available
- Refine model as new discoveries are made
"""
    else:
        report += """
The model requires **additional tuning** before use in production targeting.

**Issues to Address**:
- Some known deposits not captured in High zone
- Feature engineering may need adjustment
- Additional data sources may be needed (e.g., geochemistry)

**Recommended Actions**:
- Analyze failed detections in detail
- Adjust feature weights
- Retrain classifier with refined features
- Re-validate
"""

    report += """

---

## References

1. USGS Mineral Resources Data System (MRDS)
2. Alaska Division of Geological & Geophysical Surveys
3. Landsat 8/9 Collection 2 Surface Reflectance (USGS)
4. ALOS World 3D Digital Surface Model (JAXA)

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Save report
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✓ Validation report saved: {output_file}")

    return report
```

---

## Phase 6: Integration & Testing (Days 16-18)

### Objective: End-to-end workflow that's robust and reproducible

### 6.1 Main Execution Pipeline (Day 16)

```python
# main.py

def main():
    """
    Production pipeline for Mat-Su lode gold prospectivity.
    """
    print("="*70)
    print("MAT-SU LODE GOLD PROSPECTIVITY SYSTEM")
    print("="*70)

    # Configuration
    config = load_config('config.py')

    # Initialize Earth Engine
    ee.Initialize(project='ee-markjamesp')
    print("✓ Google Earth Engine initialized")

    # Phase 1: Data Acquisition
    print("\nPhase 1: Data Acquisition")
    print("-" * 70)

    # User validation dataset
    user_mines_gdf = load_user_validation_data(config['user_mines_file'])
    print(f"✓ Loaded {len(user_mines_gdf)} user validation mines")

    # USGS MRDS
    usgs_mines_gdf = load_usgs_mrds_data(config['usgs_mrds_file'], config['region'])
    print(f"✓ Loaded {len(usgs_mines_gdf)} USGS lode gold occurrences")

    # Satellite data
    print("\nDownloading satellite data...")
    landsat_composite = get_optimized_landsat_composite(config['region_ee'])
    print("✓ Landsat composite created")

    dem = get_optimized_dem(config['region_ee'])
    print("✓ DEM loaded")

    aster_data, aster_available = get_aster_data_if_available(config['region_ee'])
    if aster_available:
        print("✓ ASTER data loaded")
    else:
        print("⚠ ASTER not available - using Landsat only")

    # Phase 2: Feature Engineering
    print("\nPhase 2: Feature Engineering")
    print("-" * 70)

    # Spectral indices
    spectral_indices = calculate_willow_specific_indices(landsat_composite)
    print("✓ Spectral indices calculated")

    # Normalize
    spectral_normalized = normalize_all_indices(spectral_indices, config['region_ee'])
    print("✓ Indices normalized")

    # Structural features
    structural_features = detect_structural_features_optimized(dem, config['region_ee'])
    print("✓ Structural features calculated")

    # Feature stack
    feature_stack, feature_names = create_feature_stack(
        spectral_normalized,
        structural_features,
        config['region_ee']
    )
    print(f"✓ Feature stack created: {len(feature_names)} features")

    # Phase 3: Training Data Preparation
    print("\nPhase 3: Training Data Preparation")
    print("-" * 70)

    positive_samples, negative_samples = prepare_training_dataset(
        user_mines_gdf,
        usgs_mines_gdf,
        config['region']
    )
    print(f"✓ Training data prepared: {len(positive_samples)} positive, {len(negative_samples)} negative")

    # Extract features
    train, test, training_data = extract_training_features(
        feature_stack,
        positive_samples,
        negative_samples
    )
    print(f"✓ Features extracted: {train.size().getInfo()} train, {test.size().getInfo()} test")

    # Phase 4: Model Training
    print("\nPhase 4: Model Training")
    print("-" * 70)

    classifier, best_params, train_accuracy = train_optimized_random_forest(
        train,
        test,
        feature_names
    )
    print(f"✓ Model trained: Accuracy {train_accuracy:.1%}, Params {best_params}")

    # Phase 5: Validation (Critical)
    print("\nPhase 5: Model Validation")
    print("-" * 70)

    # Classify entire area
    ml_probability = feature_stack.classify(classifier.setOutputMode('PROBABILITY'))

    # Validate against user's mines
    validation_passed, mine_results = validate_model_against_user_mines(
        classifier,
        feature_stack,
        user_mines_gdf
    )

    if not validation_passed:
        print("\n⚠ VALIDATION FAILED - Entering debug mode")
        debug_failed_detections(mine_results, feature_stack, user_mines_gdf)

        # User decision point
        response = input("\nContinue with current model (c), Retune (r), or Abort (a)? ")
        if response.lower() == 'a':
            print("Aborting - model not ready for production")
            return
        elif response.lower() == 'r':
            print("Retuning model...")
            # Implement retuning logic
            # ...

    # Phase 6: Prospectivity Model
    print("\nPhase 6: Prospectivity Model Integration")
    print("-" * 70)

    prospectivity = create_optimized_prospectivity_model(
        spectral_normalized,
        structural_features,
        ml_probability,
        config['region_ee']
    )
    print("✓ Prospectivity model created")

    # Calibrate thresholds
    known_deposits_fc = gdf_to_ee_fc(
        pd.concat([user_mines_gdf, usgs_mines_gdf])
    )
    prospectivity_classified, threshold_high, threshold_medium = calibrate_thresholds(
        prospectivity,
        config['region_ee'],
        known_deposits_fc
    )
    print(f"✓ Thresholds calibrated: High≥{threshold_high:.2f}, Medium≥{threshold_medium:.2f}")

    # Phase 7: Output Generation
    print("\nPhase 7: Output Generation")
    print("-" * 70)

    # Interactive map
    map_html = create_production_map(
        config['region'],
        landsat_composite,
        spectral_normalized,
        structural_features,
        prospectivity,
        usgs_mines_gdf,
        user_mines_gdf
    )
    map_file = 'outputs/maps/matsu_willow_prospectivity.html'
    map_html.save(map_file)
    print(f"✓ Interactive map saved: {map_file}")

    # Target list
    targets_df = extract_top_targets_optimized(
        prospectivity,
        feature_stack,
        config['region_ee'],
        num_targets=100,
        min_threshold=threshold_high
    )
    targets_file = 'outputs/targets/matsu_top_100_targets.csv'
    targets_df.to_csv(targets_file, index=False)
    print(f"✓ Target list saved: {targets_file} ({len(targets_df)} targets)")

    # KML export
    kml_file = 'outputs/targets/matsu_targets.kml'
    generate_kml_output(targets_df, kml_file)
    print(f"✓ KML file saved: {kml_file}")

    # Validation report
    report_file = 'outputs/validation/validation_report.md'
    generate_validation_report(
        {
            'success_rate_high': success_rate_high,
            'mine_results': mine_results,
            # ... other metrics
        },
        model_metrics,
        feature_importance,
        report_file
    )
    print(f"✓ Validation report saved: {report_file}")

    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - Interactive map: {map_file}")
    print(f"  - Target list: {targets_file}")
    print(f"  - KML file: {kml_file}")
    print(f"  - Validation report: {report_file}")
    print(f"\nValidation: {'PASSED ✓' if validation_passed else 'FAILED ✗'}")
    print(f"Top Targets: {len(targets_df)}")
    print(f"\nNext steps: Open {map_file} in browser to explore results")
    print("="*70)

if __name__ == "__main__":
    main()
```

---

## Phase 7: Deployment & Documentation (Days 19-21)

### 7.1 Configuration File (Day 19)

```python
# config.py

"""
Configuration for Mat-Su Lode Gold Prospectivity System
"""

import ee

# Study Area
# Willow Creek Mining District core area
REGION_NAME = "Mat-Su Willow Core"
REGION_BOUNDS = [-149.50, 61.70, -149.00, 62.00]  # [west, south, east, north]
REGION_EE = ee.Geometry.Rectangle(REGION_BOUNDS)

# Data Files
USER_MINES_FILE = 'data/user_validation_mines.csv'  # User provides this
USGS_MRDS_FILE = 'data/usgs_mrds_alaska_lode.geojson'  # Downloaded by script

# Earth Engine
GEE_PROJECT = 'ee-markjamesp'

# Processing Parameters
LANDSAT_YEARS = [2020, 2021, 2022, 2023, 2024]
LANDSAT_MONTHS = [7, 8, 9]  # July-September (late summer)
MAX_CLOUD_COVER = 30  # percent

# Feature Thresholds
VEGETATION_THRESHOLD = 0.4  # NDVI > 0.4 = dense vegetation (mask)
SLOPE_THRESHOLD = 15  # degrees (< 15° may be alluvium)

# Model Parameters
RF_N_TREES = 500
RF_MIN_LEAF = 5
RF_VARS_PER_SPLIT = None  # Auto

# Validation
MIN_SUCCESS_RATE = 0.80  # 80% of known mines must be detected

# Output
NUM_TOP_TARGETS = 100
OUTPUT_DIR = 'outputs'

# Known Mines for Additional Validation (if user doesn't provide)
KNOWN_MINES_FALLBACK = [
    {'name': 'Independence Mine', 'lat': 61.7872, 'lon': -149.2736},
    {'name': 'Lucky Shot Mine', 'lat': 61.7945, 'lon': -149.2891},
    {'name': 'Fern Mine', 'lat': 61.7823, 'lon': -149.2654},
    {'name': 'Gold Cord Mine', 'lat': 61.7912, 'lon': -149.2823},
    {'name': 'War Baby Mine', 'lat': 61.7756, 'lon': -149.2598}
]
```

### 7.2 README.md (Day 19)

```markdown
# Mat-Su Lode Gold Prospectivity System

Production-grade gold exploration tool for the Matanuska-Susitna Valley (Willow Creek District), Alaska.

## Features

✓ Multi-source data integration (Landsat, ALOS DEM, USGS MRDS)
✓ Machine learning classification (Random Forest)
✓ Structural analysis (fault/shear detection)
✓ Validated against known successful mines
✓ Interactive HTML maps
✓ Ranked target list (top 100 anomalies)
✓ KML export for Google Earth

## Requirements

- Python 3.11
- Google Earth Engine account (authenticated)
- Packages: earthengine-api, geopandas, rasterio, folium, scikit-learn

## Installation

1. Clone repository
2. Install packages: `pip install -r requirements.txt`
3. Authenticate GEE: `python -c "import ee; ee.Authenticate()"`
4. Provide validation data: Save your mine locations to `data/user_validation_mines.csv`

## Usage

### Basic Run

```bash
python main.py
```

This will:
1. Download satellite data for Mat-Su Valley
2. Calculate mineral and structural indices
3. Train machine learning model
4. Validate against your mine data
5. Generate prospectivity map
6. Export top 100 targets

### Configuration

Edit `config.py` to adjust:
- Study area bounds
- Processing parameters
- Model hyperparameters
- Output settings

### Outputs

All outputs saved to `outputs/` directory:

- **Interactive Map**: `outputs/maps/matsu_willow_prospectivity.html`
  - Open in web browser
  - Toggle layers, zoom to targets

- **Target List**: `outputs/targets/matsu_top_100_targets.csv`
  - Ranked anomalies with coordinates
  - Feature values for each target

- **KML File**: `outputs/targets/matsu_targets.kml`
  - Import to Google Earth Pro
  - Color-coded by prospectivity

- **Validation Report**: `outputs/validation/validation_report.md`
  - Model performance metrics
  - Known mine detection results
  - Feature importance

## Validation Data Format

Your mine locations file should be CSV with these columns:

| Name | Latitude | Longitude |
|------|----------|-----------|
| Independence Mine | 61.7872 | -149.2736 |
| Lucky Shot Mine | 61.7945 | -149.2891 |
| ... | ... | ... |

Optional columns: Production, Status, Notes

## Interpreting Results

### Prospectivity Scores

- **High (>0.7)**: Priority targets for field work
- **Medium (0.5-0.7)**: Secondary targets
- **Low (<0.5)**: Background

### Target Prioritization

Top targets show combination of:
- Strong iron oxide (gossan from weathered sulfides)
- Carbonate alteration (diagnostic for Willow deposits)
- North-trending lineaments (structural control)
- Proximity to known deposits

### Field Verification

For each target:
1. Check land ownership/access
2. Conduct reconnaissance
3. Sample rock chips (assay for Au, As, Sb)
4. Map structures and alteration
5. Consider geophysical survey

## Troubleshooting

**"Validation failed" message**
- Model couldn't detect enough known mines
- Check if your mine coordinates are correct
- May need to adjust feature weights
- Run debug mode for details

**"No ASTER data available"**
- Normal for Alaska (spotty coverage)
- Script will use Landsat-only indices
- Still produces good results

**"Timeout" errors**
- Study area may be too large
- Reduce REGION_BOUNDS in config.py
- Or process in multiple runs

## Model Limitations

- Vegetation obscures geology (focus on alpine areas)
- Glacial cover limits high-elevation detection
- Trained on Willow-type deposits (may not detect other types)
- Requires validation against known mines

## Citation

If using this tool for research or commercial work:

```
Mat-Su Lode Gold Prospectivity System
Willow Creek Mining District, Alaska
Version 1.0 (2025)
```

## License

[Your chosen license]

## Contact

[Your contact information]
```

### 7.3 requirements.txt (Day 19)

```
earthengine-api>=0.1.350
geopandas>=0.12.0
rasterio>=1.3.0
folium>=0.14.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
simplekml>=1.3.6
shapely>=2.0.0
```

---

## Success Metrics & Quality Assurance (Day 20-21)

### Final Checklist

**Data Quality**:
- [ ] User validation dataset loaded correctly
- [ ] USGS MRDS data filtered for lode only (no placer)
- [ ] Study area contains known deposits
- [ ] Satellite composites are cloud/snow-free

**Model Performance**:
- [ ] Validation success rate ≥80%
- [ ] Independence Mine detected as HIGH
- [ ] ≤5% of study area classified as HIGH
- [ ] Feature importance makes geological sense

**Output Quality**:
- [ ] Interactive map loads in browser
- [ ] All layers toggle correctly
- [ ] Known mines visible on map
- [ ] CSV opens in Excel, coordinates correct
- [ ] KML displays in Google Earth
- [ ] Validation report complete and readable

**Code Quality**:
- [ ] All functions documented
- [ ] Error handling implemented
- [ ] Progress indicators working
- [ ] Reproducible (same inputs → same outputs)

**Production Readiness**:
- [ ] README complete
- [ ] Configuration file documented
- [ ] Example outputs provided
- [ ] Troubleshooting guide written

---

## Timeline Summary

| Week | Days | Phase | Deliverable |
|------|------|-------|-------------|
| 1 | 1-2 | Assessment & Setup | Baseline metrics, data acquired |
| 1 | 3-5 | Data Pipeline | Optimized composites, training data |
| 2 | 6-8 | Feature Engineering | Calibrated indices, structural features |
| 2 | 9-11 | Machine Learning | Trained model, validation passed |
| 3 | 12-13 | Prospectivity Model | Final prospectivity map |
| 3 | 14-15 | Outputs | Maps, CSV, KML, report |
| 3 | 16-18 | Integration & Testing | End-to-end pipeline working |
| 3 | 19-21 | Documentation & QA | Production-ready system |

---

## Critical Success Factors

1. **User Validation Data** - Must have accurate mine locations
2. **Proper Normalization** - Fixes "everything is red" problem
3. **Structural Integration** - Willow deposits are fault-controlled
4. **Validation Loop** - Iterate until model passes
5. **Clear Outputs** - Actionable target list

## Risk Mitigation

**Risk**: User doesn't have validation data
- **Mitigation**: Use USGS MRDS only, acknowledge lower confidence

**Risk**: Model doesn't validate (< 80%)
- **Mitigation**: Debug protocol in place, feature analysis tools

**Risk**: ASTER data not available
- **Mitigation**: Landsat-only indices optimized, still effective

**Risk**: Processing timeouts
- **Mitigation**: Reduce study area, use smaller time windows

---

## Post-Deployment Plan

### Phase 8: Field Validation (Ongoing)

As user visits targets:
1. Record GPS coordinates
2. Photograph outcrops/alteration
3. Collect rock chip samples
4. Assay for Au, As, Sb (pathfinders)
5. Update model with results

### Phase 9: Model Refinement

After field season:
1. Add confirmed deposits to training data
2. Add false positives as negative samples
3. Retrain classifier
4. Validate against new data
5. Release updated model

### Phase 10: Expansion

Apply model to:
1. Adjacent areas in Mat-Su Valley
2. Other orogenic gold districts in Alaska
3. Similar terranes worldwide

---

**END OF OPTIMIZATION PLAN**

This plan transforms the basic script into a production-ready exploration tool validated against real successful mines.
