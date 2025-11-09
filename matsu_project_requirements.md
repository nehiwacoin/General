# Mat-Su Valley Lode Gold Prospectivity System - Project Requirements

## Target Area: Matanuska-Susitna Valley (Willow Mining District), Alaska

### Geographic Focus

**Primary Study Area: Willow Creek Mining District**
- Latitude: 61.7° N to 62.0° N
- Longitude: -149.5° W to -149.0° W
- Size: Approximately 50 km × 30 km
- Elevation: 500m - 2000m

**Known Lode Deposits in Area (Validation Sites):**
- Independence Mine (historic major producer)
- Lucky Shot Mine
- Fern Mine
- Gold Cord Mine
- War Baby Mine
- Mabel Mine
- Thorpe Mine
- Martin Mine
- Lonesome Mine

**User's Validation Data:**
- User has data on successful mine sites in the area
- This will be the PRIMARY validation dataset
- Model must detect these known successful operations

---

## Geological Setting - Mat-Su Lode Gold Deposits

### Regional Geology

**Host Rocks:**
- Willow Creek Schist (Jurassic-Cretaceous)
- Greenschist to amphibolite facies metamorphics
- Interbedded metasediments and metavolcanics
- Quartz-mica schist dominant

**Tectonic Setting:**
- Accretionary terrane (Peninsular Terrane)
- Major regional thrust faults
- North-trending fault systems
- Compressional deformation

**Intrusive Bodies:**
- Cretaceous granitic plutons nearby
- Contact metamorphic aureoles
- Potential heat/fluid source for mineralization

### Deposit Type: Orogenic Lode Gold

**Characteristics:**
- **Type**: Mesothermal orogenic vein deposits
- **Age**: Late Cretaceous (~70 Ma)
- **Structural Control**: Shear zones, faults, fold hinges
- **Ore Mineralogy**:
  - Native gold (high fineness, 800-900)
  - Pyrite (dominant sulfide)
  - Arsenopyrite (key pathfinder)
  - Stibnite (antimony sulfide - diagnostic)
  - Minor chalcopyrite, galena, sphalerite

**Vein Characteristics:**
- Quartz veins (1 cm to several meters wide)
- En echelon vein arrays
- Multiple generations of veining
- Vein swarms at structural junctions

**Alteration Assemblages:**

1. **Proximal (at vein):**
   - Silicification (quartz flooding)
   - Sericite (white mica alteration)
   - Carbonate (ankerite, siderite, calcite)
   - Chlorite

2. **Distal (alteration halo):**
   - Chlorite-sericite schist
   - Carbonate alteration
   - Disseminated pyrite

3. **Weathering Products (surface expression):**
   - Iron oxide gossan (limonite, goethite)
   - Clay minerals (from sericite breakdown)
   - Ochre staining on outcrops
   - Boxwork textures (after pyrite/arsenopyrite)

**Structural Controls (CRITICAL):**
- North to NNE trending faults (dominant orientation)
- Fault intersections (prime targets)
- Shear zones in competent schist units
- Fold hinge zones
- Dilatational jogs in fault systems

---

## Remote Sensing Signatures - Mat-Su Specific

### Spectral Signatures to Target

**1. Iron Oxide (Gossan) - PRIMARY INDICATOR**
- **What**: Weathered sulfides (pyrite → limonite/goethite)
- **Signature**: Fe³⁺ absorption at ~0.87 µm
- **Landsat Detection**: High Red/Blue ratio (B4/B2)
- **Appearance**: Rusty orange/brown outcrops
- **Significance**: Direct surface expression of mineralized veins
- **Challenge**: Alaska has abundant glacial sediments - need to distinguish bedrock from transported material

**2. Carbonate Alteration - DIAGNOSTIC FOR WILLOW**
- **What**: Ankerite, calcite, siderite (CO₃ minerals)
- **Signature**: CO₃²⁻ absorption features in SWIR
- **ASTER Detection**: TIR bands show carbonate signature
- **Significance**: Strongly associated with Willow-type deposits
- **Unique to orogenic gold**: Distinguishes from other deposit types

**3. Sericite/Phyllic Alteration**
- **What**: Fine-grained white mica (muscovite/sericite)
- **Signature**: Al-OH absorption at 2.2 µm
- **ASTER Detection**: Better than Landsat (dedicated SWIR bands)
- **Landsat Proxy**: SWIR1/SWIR2 ratio (B6/B7)
- **Significance**: Core alteration zone

**4. Chlorite Alteration**
- **What**: Mg-Fe chlorite (propylitic alteration)
- **Signature**: Fe-Mg-OH absorption features
- **Detection**: SWIR bands
- **Significance**: Distal alteration, broader target

**5. Silicification**
- **What**: Quartz veining, silica flooding
- **Signature**: High reflectance, low Fe absorption
- **Detection**: Brightness index, TIR characteristics
- **Significance**: Actual mineralized structure

### Challenges Specific to Alaska

**1. Vegetation Cover**
- Dense boreal forest (spruce, birch)
- Obscures bedrock in valleys
- Use NDVI to mask vegetated areas
- Focus on alpine/subalpine exposures (>800m elevation)
- Ridge crests often vegetation-free

**2. Snow and Ice**
- Persistent snowpack (even summer images)
- Glacial cover at high elevations
- Need strict cloud/snow masking
- Use late summer imagery (August-September)

**3. Glacial Deposits**
- Extensive till, moraines, outwash
- Can transport mineralized material (placer gold)
- Need to distinguish bedrock vs. transported gossan
- Use slope/topography filters (bedrock on steep slopes)

**4. Short Field Season**
- Limited cloud-free imagery
- May need multi-year composites
- Quality over quantity

**5. Permafrost Effects**
- Periglacial features (solifluction)
- Rock glaciers
- Can obscure or enhance structural expression

---

## Data Sources and Availability

### Satellite Data (Google Earth Engine)

**Tier 1 - Primary Data (Confirmed Available):**

1. **Landsat 8/9 Surface Reflectance**
   - GEE: `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`
   - Resolution: 30m
   - Bands: Blue, Green, Red, NIR, SWIR1, SWIR2, TIR
   - Coverage: Excellent (path/row overlaps)
   - Date Range: 2013-present (Landsat 8), 2021-present (Landsat 9)
   - **Use**: Primary mineral detection

2. **SRTM DEM**
   - GEE: `USGS/SRTMGL1_003`
   - Resolution: 30m
   - Coverage: Good for Mat-Su (< 60°N generally good)
   - **Issue**: May have voids in Alaska
   - **Use**: Structural analysis, slope, aspect

3. **ALOS World 3D DEM**
   - GEE: `JAXA/ALOS/AW3D30/V3_2`
   - Resolution: 30m
   - Coverage: Excellent for Alaska
   - **Better quality than SRTM for high latitudes**
   - **Use**: Primary DEM source

**Tier 2 - Enhanced Data:**

4. **ASTER L1T**
   - GEE: `ASTER/AST_L1T_003`
   - VNIR: 3 bands (15m)
   - SWIR: 6 bands (30m) - excellent for alteration
   - TIR: 5 bands (90m) - carbonate detection
   - Coverage: Spotty for Alaska (on-demand acquisition)
   - **Check availability for study area**
   - **Use**: Advanced alteration mapping

5. **Sentinel-2**
   - GEE: `COPERNICUS/S2_SR`
   - Resolution: 10-20m
   - More frequent acquisitions
   - Red-edge bands useful
   - **Use**: Supplement Landsat

### External Datasets

**1. USGS MRDS (Mineral Resources Data System)**
- URL: https://mrdata.usgs.gov/mrds/
- Format: CSV, Shapefile
- Filter Criteria:
  - State: Alaska
  - Commodity: Gold (Au)
  - Deposit Type: "Lode" (NOT placer)
  - Region: Willow Creek district
- **Known deposits**: ~30-40 lode gold occurrences in district
- **Use**: Training points, validation

**2. Alaska Division of Geological & Geophysical Surveys (DGGS)**
- Geological maps (bedrock geology)
- Geochemical surveys (if available)
- Structural data (fault maps)
- **Check availability for Willow Creek area**

**3. USGS Alaska Resource Data Files (ARDF)**
- Detailed mine descriptions
- Production data
- Ore characteristics
- **Use**: Understanding deposit characteristics

**4. User's Validation Dataset**
- **CRITICAL**: User has data on successful mine sites
- Format: TBD (lat/lon coordinates? shapefile?)
- **This is the PRIMARY validation source**
- Model MUST detect these sites

**5. Alaska Geologic Materials Center**
- Core samples, assay data (if accessible)
- **Optional**: High-confidence validation

---

## Structural Analysis Requirements

### Critical Structural Features for Willow-Type Deposits

**1. Fault/Shear Zone Detection**
- **Orientation**: North to NNE trending (dominant)
- **Detection Method**:
  - Lineament extraction from hillshade (multiple azimuths)
  - Edge detection on DEM derivatives
  - Sobel/Canny filters
- **Output**: Lineament density map, fault traces

**2. Fault Intersections**
- **Why Critical**: Willow deposits cluster at fault junctions
- **Detection**:
  - Focal sum of lineaments
  - Areas with multiple intersecting structures
- **Output**: Intersection density map

**3. Topographic Position**
- **Ridge Crests**: Vein exposures (resistant quartz)
- **Steep Slopes**: Fault scarps
- **Structural Highs**: Fold culminations
- **Detection**:
  - Topographic Position Index (TPI)
  - Slope >30° (bedrock exposure)
  - Ridge/valley classification

**4. Curvature Analysis**
- **Plan Curvature**: Detects lineaments (cross-slope)
- **Profile Curvature**: Ridges vs valleys
- **Use**: Enhance structural complexity

**5. Drainage Patterns**
- **Rectilinear Drainage**: Fault-controlled
- **Offset Streams**: Active faults
- **Detection**: Stream network analysis

---

## Machine Learning Approach

### Training Dataset Construction

**Positive Class (Lode Gold Occurrences):**

1. **User's Validation Data** (PRIMARY)
   - Successful mine sites in Mat-Su
   - High confidence
   - Known producers

2. **USGS MRDS Lode Gold**
   - Filter for Willow Creek district
   - Exclude placer deposits
   - Include: Producer, Past Producer, Prospect, Occurrence
   - Expected: ~30-40 points

3. **Manual Additions** (if needed)
   - Known but undocumented prospects
   - User's local knowledge

**Negative Class (Background):**
- Random points in study area
- **Exclusion buffer**: 2-5 km from known deposits
- **Stratified sampling**: Different elevations, rock types
- **Number**: 2-3× positive samples (balance dataset)

### Feature Vector (per point)

**Spectral Features (Landsat 8/9):**
1. Band 2 (Blue) - mean value
2. Band 3 (Green)
3. Band 4 (Red)
4. Band 5 (NIR)
5. Band 6 (SWIR1)
6. Band 7 (SWIR2)
7. Iron oxide ratio (B4/B2)
8. Clay minerals ratio (B6/B7)
9. Ferrous minerals ratio (B6/B5)
10. Carbonate proxy (B7/B6)
11. Brightness index
12. NDVI

**Spectral Features (ASTER - if available):**
13. AlOH index
14. FeOH index
15. Carbonate index (TIR)
16. Silica index (TIR)

**Structural Features (DEM):**
17. Elevation
18. Slope
19. Aspect
20. Plan curvature
21. Profile curvature
22. TPI (Topographic Position Index)
23. TRI (Terrain Ruggedness Index)
24. Distance to nearest lineament
25. Lineament density (1 km buffer)
26. Fault intersection proximity
27. Drainage density

**Proximity Features:**
28. Distance to nearest known deposit
29. Deposit density (5 km buffer)
30. Distance to intrusive contact (if geological map available)

**Total Features**: ~30 (Landsat only) or ~34 (with ASTER)

### Classification Algorithm

**Primary**: Random Forest
- Trees: 500
- Min leaf population: 5
- Train/Test split: 70/30
- Cross-validation: 5-fold

**Why Random Forest**:
- Handles non-linear relationships (geology is non-linear)
- Robust to outliers
- Provides feature importance
- No overfitting with enough trees
- Interpretable

**Alternative** (if RF underperforms):
- Gradient Boosting (XGBoost)
- Support Vector Machine (SVM)

### Validation Strategy

**1. Cross-Validation**
- K-fold (k=5)
- Stratified (maintain class balance)

**2. Known Mine Validation**
- Hold out user's successful mine sites
- Predict on these
- Success rate: % detected in "High" zones
- **Target**: ≥80% capture rate

**3. Spatial Validation**
- Train on one area, test on another
- Check model generalization

**4. Metrics**
- Confusion Matrix
- Accuracy, Precision, Recall
- AUC-ROC
- Feature importance ranking

---

## Prospectivity Model Integration

### Multi-Criteria Decision Analysis (MCDA)

**Evidence Layers** (weighted combination):

1. **Spectral/Alteration Evidence** (35%)
   - Iron oxide anomalies (normalized)
   - Carbonate alteration
   - Sericite alteration
   - Composite alteration index

2. **Structural Evidence** (40%)
   - Lineament density
   - Fault intersection proximity
   - North-trending lineament density (Willow-specific)
   - Structural complexity

3. **Machine Learning Prediction** (20%)
   - Random Forest probability output
   - Integrated spectral + structural learning

4. **Proximity/Clustering** (5%)
   - Distance to known deposits
   - Clustering analysis
   - "Near mine" bonus

**Formula**:
```
Prospectivity = 0.35 × Spectral_Score +
                0.40 × Structural_Score +
                0.20 × ML_Probability +
                0.05 × Proximity_Score
```

**Normalization**:
- Each layer normalized to 0-1 using 5th-95th percentile
- Prevents one layer from dominating

**Masking**:
- Mask dense vegetation (NDVI > 0.4)
- Mask glaciers/snow
- Mask water bodies
- Focus on bedrock exposures

### Prospectivity Classification

**Thresholds** (calibrated to known deposits):
- **High**: >0.7 (top 5% of area) - Priority field targets
- **Medium**: 0.5-0.7 (next 10%) - Secondary targets
- **Low**: <0.5 (background) - Not recommended

**Validation**:
- Check what % of known deposits fall in "High" zones
- If <80%, adjust weights and rerun
- Iterate until validated

---

## Validation Against Known Mines

### Primary Validation Sites (User's Data)

**User must provide**:
- Mine names
- Coordinates (lat/lon or UTM)
- Production status (if known)
- Any notes on ore characteristics

**Expected mines in dataset** (if user has these):
- Independence Mine
- Lucky Shot
- Fern Mine
- Gold Cord
- War Baby
- Mabel
- Thorpe
- Others user specifies

### Validation Process

**For each mine**:
1. Extract prospectivity score at mine location
2. Classify as High/Medium/Low
3. Compare to expected (all should be High)
4. Calculate success rate

**Success Criteria**:
- ≥80% of known producers → High prospectivity
- ≥90% of known producers → Medium or High
- Independence Mine (major producer) → MUST be High

**If validation fails**:
- Analyze which features are weak at failed sites
- Adjust feature weights
- Retrain classifier
- Iterate

---

## Expected Outputs

### 1. Interactive HTML Map

**Layers**:
- True color satellite imagery (Landsat)
- Iron oxide anomalies
- Carbonate/clay alteration
- Structural lineaments
- Prospectivity zones (color-coded)
- Known deposits (validation points)
- User's mine sites (highlighted)

**Features**:
- Toggle layers on/off
- Zoom to high-priority targets
- Click on anomalies for details
- Export view as image

### 2. Target List (CSV)

**Top 100 ranked anomalies**:

| Rank | Latitude | Longitude | Prospectivity | Iron_Oxide | Carbonate | Lineament_Density | Distance_to_Known | Notes |
|------|----------|-----------|---------------|------------|-----------|-------------------|-------------------|-------|
| 1    | 61.7850  | -149.2341 | 0.87          | 0.92       | 0.85      | 0.78              | 0.5 km            | Near Independence |
| 2    | 61.7923  | -149.3012 | 0.84          | 0.88       | 0.80      | 0.82              | 1.2 km            | Fault intersection |
| ...  | ...      | ...       | ...           | ...        | ...       | ...               | ...               | ... |

**Additional columns**:
- UTM coordinates
- Elevation
- Slope
- Aspect
- Nearest known mine
- Confidence level

### 3. KML/KMZ File (Google Earth)

**Color-coded points**:
- Red: High prospectivity (>0.7)
- Orange: Medium (0.5-0.7)
- Yellow: Low-medium (0.3-0.5)

**Includes**:
- Top 100 targets
- Known deposits (validation)
- User's mine sites (flagged)

### 4. Validation Report (Markdown/PDF)

**Contents**:
1. Executive Summary
   - Study area
   - Number of targets identified
   - Validation success rate

2. Methodology
   - Data sources used
   - Features calculated
   - ML algorithm details

3. Results
   - Prospectivity map (image)
   - Top 20 targets (table)
   - Known mine validation (table)

4. Validation Statistics
   - Confusion matrix
   - ROC curve
   - Feature importance plot
   - Known mine capture rate

5. Recommendations
   - Priority targets for field work
   - Areas needing ground verification
   - Suggested follow-up work

### 5. GeoTIFF Rasters (for GIS)

**Exportable layers**:
- Prospectivity map (0-1 values)
- Iron oxide index
- Carbonate alteration
- Lineament density
- ML probability

**Format**: GeoTIFF, WGS84 (EPSG:4326)
**Use**: Import to QGIS/ArcGIS for detailed analysis

---

## Technical Requirements

### Software Environment

**Python Version**: 3.11 (already installed)

**Required Packages** (already installed):
- earthengine-api (GEE Python API)
- geopandas (spatial data handling)
- rasterio (raster I/O)
- folium (interactive maps)
- pandas (data manipulation)
- numpy (numerical operations)
- scikit-learn (machine learning)
- matplotlib (plotting)
- simplekml (KML generation)

**Google Earth Engine**:
- Already authenticated
- Project: `ee-markjamesp`

### Computational Considerations

**GEE Limits**:
- Export size: <50 MB per export
- Computation timeout: May need to chunk large areas
- Server-side processing: Most operations on GEE servers

**Local Processing**:
- Feature extraction: Client-side (from GEE)
- ML training: Local (scikit-learn)
- Map generation: Local (folium)

**Processing Time Estimates**:
- Data download/composite: 5-10 min
- Feature calculation: 5-10 min
- ML training: 1-2 min
- Prospectivity map: 5-10 min
- **Total per region**: ~30-45 minutes

---

## Study Area Definition

### Recommended Bounding Box

**Option 1: Focused (Willow Creek District Core)**
```python
MATSU_WILLOW_CORE = {
    'name': 'Mat-Su Willow Core',
    'bounds': [-149.50, 61.70, -149.00, 62.00],  # [west, south, east, north]
    'description': 'Core Willow Creek mining district',
    'known_deposits': ['Independence', 'Lucky Shot', 'Fern', 'Gold Cord', 'War Baby']
}
```

**Option 2: Extended (Broader Exploration)**
```python
MATSU_WILLOW_EXTENDED = {
    'name': 'Mat-Su Willow Extended',
    'bounds': [-150.00, 61.50, -148.50, 62.20],
    'description': 'Extended Mat-Su valley exploration area',
    'known_deposits': ['All Willow district + nearby prospects']
}
```

**Option 3: User-Defined**
```python
# User can specify custom bounds based on their area of interest
MATSU_CUSTOM = {
    'name': 'Mat-Su Custom AOI',
    'bounds': [west, south, east, north],  # User provides
    'description': 'User-defined area of interest'
}
```

---

## Success Criteria (Pass/Fail)

### Model Performance

1. **Known Mine Detection**: ≥80% of user's successful mines → High prospectivity
2. **Discrimination**: ≤5% of study area classified as High
3. **Independence Mine**: MUST score High (major validation check)
4. **No False Positives**: High zones show alteration + structural evidence
5. **Actionable Targets**: Top 100 list with coordinates and justification

### Output Quality

6. **Interactive Map**: Loads in browser, layers work, readable
7. **CSV Export**: Opens in Excel, coordinates correct, sortable
8. **KML File**: Displays correctly in Google Earth
9. **Validation Report**: Professional, includes all required sections
10. **Code Quality**: Documented, modular, reproducible

---

## Development Priorities

### Phase 1: MVP (Week 1)
**Goal**: Working model that validates against known deposits

1. Define study area (user's preferred bounds)
2. Download USGS MRDS data (lode gold in Alaska)
3. Integrate user's validation dataset
4. Landsat composite (2023-2024, summer only)
5. ALOS DEM derivatives
6. Basic mineral indices (iron oxide, clay, carbonate proxy)
7. Lineament detection (simple edge detection)
8. Random Forest classifier
9. Prospectivity map
10. **Validate against user's mines** ← Critical checkpoint
11. If validation passes (≥80%), proceed to Phase 2
12. If fails, debug and retune

### Phase 2: Enhancement (Week 2)
**Goal**: Improve discrimination and add features

13. Add ASTER data (if available)
14. Advanced lineament extraction (multiple azimuths)
15. Fault intersection detection
16. Carbonate alteration (TIR bands)
17. Weighted overlay optimization
18. Target ranking and filtering
19. HTML map with all layers
20. CSV and KML exports
21. Professional validation report

### Phase 3: Refinement (Optional)
**Goal**: Production-ready tool

22. Add geological map (if available)
23. Geochemical data integration
24. Ensemble ML models
25. Uncertainty quantification
26. Automated report generation
27. User interface for parameter adjustment

---

## User Action Items (Before Starting)

### Required from User:

1. **Validation Dataset**
   - File format: CSV, Excel, Shapefile, or KML
   - Required columns: Mine Name, Latitude, Longitude
   - Optional: Production status, Ore type, Notes
   - **Where to save**: `data/user_validation_mines.csv`

2. **Study Area Preference**
   - Choose Option 1 (Core), Option 2 (Extended), or Option 3 (Custom)
   - If custom, provide coordinates: [west, south, east, north]

3. **Priority Mines**
   - List 5-10 mines that MUST be detected (highest confidence)
   - These will be the primary validation checkpoints

### Optional from User:

4. **Local Knowledge**
   - Any prospects not in USGS database?
   - Specific structural features to target?
   - Areas to exclude (e.g., glaciers, water)?

5. **Deposit Characteristics**
   - Any specific ore mineralogy notes?
   - Alteration types observed?
   - Structural orientations?

---

## File Structure

```
matsu_lode_gold_project/
│
├── config.py                           # Study area, parameters
├── main.py                             # Main execution script
│
├── modules/
│   ├── __init__.py
│   ├── gee_data.py                    # GEE data download
│   ├── usgs_data.py                   # USGS MRDS download
│   ├── spectral_indices.py            # Mineral indices
│   ├── structural_analysis.py         # DEM analysis, lineaments
│   ├── ml_classifier.py               # Random Forest
│   ├── prospectivity.py               # Weighted overlay
│   ├── validation.py                  # Known mine validation
│   └── outputs.py                     # Map, CSV, KML generation
│
├── data/
│   ├── user_validation_mines.csv      # USER PROVIDES THIS
│   ├── usgs_mrds_alaska_lode.geojson  # Downloaded by script
│   └── known_mines_matsu.json         # Validation reference
│
├── outputs/
│   ├── maps/
│   │   └── matsu_willow_prospectivity.html
│   ├── targets/
│   │   ├── matsu_top_100_targets.csv
│   │   └── matsu_targets.kml
│   └── validation/
│       ├── validation_report.md
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       └── known_mine_results.csv
│
├── README.md                          # Usage instructions
├── matsu_project_requirements.md      # This file
└── requirements.txt                   # Python dependencies
```

---

## Next Steps

1. **User provides validation dataset** (mine locations)
2. **User selects study area** (Core, Extended, or Custom bounds)
3. **Start development** with Phase 1 MVP
4. **Validate against user's mines** ← Critical milestone
5. **Iterate if needed** to achieve ≥80% detection rate
6. **Proceed to Phase 2** only after validation passes

---

## Questions for User (Before Starting)

1. What format is your validation dataset? (CSV, Excel, Shapefile, KML)
2. How many successful mine sites do you have data for?
3. Do you have a preferred study area bounds, or should we use the Willow Core default?
4. Are there any specific mines that MUST be detected (highest priority)?
5. Do you have any additional datasets (geochemistry, structural data, etc.)?
6. What's your timeline for this project?

---

## References

**Geological:**
- USGS Professional Paper on Alaska gold deposits
- Alaska DGGS reports on Willow Creek district
- Academic papers on orogenic gold systems

**Remote Sensing:**
- USGS guides on mineral detection with Landsat
- ASTER alteration mapping techniques
- Alaska-specific remote sensing challenges

**Data Sources:**
- USGS MRDS: https://mrdata.usgs.gov/mrds/
- Alaska DGGS: https://dggs.alaska.gov/
- GEE Data Catalog: https://developers.google.com/earth-engine/datasets

---

**END OF REQUIREMENTS DOCUMENT**

This is a focused, actionable requirements document specific to the Mat-Su/Willow area in Alaska for lode gold prospectivity mapping.
