# Data Structure Reference

## S2-SHIPS Internal Layout

After extraction, the dataset is organized at `/data/praxis/raw/s2ships/extracted/S2SHIPS/`:

```
S2SHIPS/
├── coco-s2ships.json                     # COCO annotations v1 (143 annotations, 3 categories)
├── coco-1622202804.029818.json           # COCO annotations v2 (159 annotations, 4 categories) ← USE THIS
├── dataset_tif/                          # Multispectral TIFFs, one subdirectory per scene
│   ├── brest1/
│   │   ├── 2021-04-14-..._B01_(Raw).tiff
│   │   ├── 2021-04-14-..._B02_(Raw).tiff
│   │   ├── ... (all 12 bands)
│   │   ├── 2021-04-14-..._NDWI.tiff
│   │   ├── 2021-04-14-..._True_color.jpg
│   │   ├── 2021-04-14-..._True_color.tiff
│   │   ├── brest1_ndwi.png
│   │   ├── brest1_rgb.png                # ← Referenced by COCO annotations
│   │   ├── brest1_segm.png
│   │   └── brest1_water.tif
│   ├── marseille/
│   ├── ... (16 scene directories total)
├── dataset_npy/                          # NumPy mask versions
│   ├── 01_mask_rome.npy
│   ├── 02_mask_suez1.npy
│   ├── ... (16 mask files)
├── s2ships_labels_mask/                  # Segmentation label PNGs
│   ├── 01_mask_rome_rgb.png
│   ├── ... (16 mask images)
├── water_mask/                           # Water mask TIFFs (geo-referenced)
│   ├── brest1_water.tif
│   ├── ... (16 water masks)
├── pretrained_backbones/                 # ResNet weights from Ciocarlan paper (not used)
└── S2SHIPS.tar.xz                        # Compressed archive (5.4MB)
```

## Scene Inventory

| Scene ID | Location | Acquisition Date | Chokepoint? |
|---|---|---|---|
| brest1 | Brest, France | 2021-04-14 | No |
| marseille | Marseille, France | 2021-04-19 | No |
| panama | Panama Canal entrance (Colon) | 2021-04-22 | **Yes** |
| portsmouth | Portsmouth, UK | 2021-04-26 | No |
| rome | Rome, Italy (Ostia) | 2021-04-30 | No |
| rotterdam1 | Rotterdam, Netherlands | 2021-04-20 | No |
| rotterdam2 | Rotterdam, Netherlands | 2021-04-20 | No |
| rotterdam3 | Rotterdam, Netherlands | 2021-04-05 | No |
| southampton | Southampton, UK | 2021-04-26 | No |
| suez1 | Suez Canal (north entrance) | 2021-04-28 | **Yes** |
| suez2 | Suez Canal (upper) | 2021-04-28 | **Yes** |
| suez3 | Suez Canal (mid) | 2021-04-28 | **Yes** |
| suez4 | Suez Canal (mid-lower) | 2021-04-28 | **Yes** |
| suez5 | Suez Canal (Great Bitter Lake area) | 2021-04-28 | **Yes** |
| suez6 | Suez Canal (Red Sea exit) | 2021-04-28 | **Yes** |
| toulon | Toulon, France | 2021-04-24 | No |

**Note:** The six Suez scenes are spatially adjacent tiles from a single Sentinel-2 acquisition on 2021-04-28. They are not statistically independent. See `methodological_decisions.md`.

## Image Properties

All TIFF imagery shares these properties:

- **Dimensions:** 938 × 1783 pixels
- **Resolution:** ~10 m × 10 m (B05 and B11 upsampled from native 20m)
- **CRS:** EPSG:3857 (Web Mercator)
- **Dtype:** uint16 (12-bit data scaled to 16-bit container)
- **Product:** Sentinel-2 L2A (bottom-of-atmosphere, atmospherically corrected)

## COCO Annotation Format

The praxis uses `coco-1622202804.029818.json` as the canonical annotation source. Key structural facts:

### Top-level keys

```json
{
  "images": [...],         // 16 entries
  "categories": [...],     // 4 entries
  "annotations": [...]     // 159 entries
}
```

### Categories

| ID | Name | Use in Praxis |
|---|---|---|
| 1 | ship | INCLUDED (positive class) |
| 2 | marina | EXCLUDED (dense recreational clusters) |
| 3 | doubt | EXCLUDED (annotator uncertain) |
| 4 | moored ship | INCLUDED (positive class) |

### Annotation grouping (critical)

**Each annotation record may contain multiple per-vessel polygons.** The COCO `bbox` field on an annotation is the bounding rectangle of ALL polygons in that annotation combined, NOT the per-vessel bounding box. The `isbbox: false` flag confirms this.

Example annotation:
```json
{
  "id": 92,
  "image_id": 1,
  "category_id": 1,
  "segmentation": [
    [390.2, 58.7, 392.2, 56.7, 394.2, 58.7, 392.2, 60.7],  // vessel 1
    [486.5, 151.0, 488.5, 149.0, 490.5, 151.0, 488.5, 153.0],  // vessel 2
    // ... 5 more polygons (7 vessels total in this one annotation)
  ],
  "area": 56,
  "bbox": [390.0, 1.0, 218.0, 351.0],  // bounds of ALL 7 polygons combined
  "isbbox": false
}
```

To obtain per-vessel bounding boxes, each polygon in `segmentation` must be processed independently: compute the axis-aligned bounding box of the polygon vertices. This "explosion" is implemented in `scripts/explode_coco_annotations.py` (to be created).

### Image references

COCO annotations reference the **RGB PNG preview** of each scene (e.g., `brest1_rgb.png`), not the multispectral TIFFs. The PNG and TIFF for each scene have identical dimensions (938 × 1783), so pixel coordinates are assumed to be 1:1 mappable. This assumption is verified in the explosion script.

## Vessel Polygon Counts

After parsing the canonical COCO file with the POSITIVE_CATEGORIES filter (ship + moored ship):

| Scene | ship | moored | Total positive | Marina (excluded) | Doubt (excluded) |
|---|---|---|---|---|---|
| brest1 | 113 | 33 | 146 | 7 | 0 |
| marseille | 38 | 27 | 65 | 44 | 1 |
| panama | 55 | 14 | 69 | 7 | 0 |
| portsmouth | 42 | 0 | 42 | 34 | 11 |
| rome | 59 | 3 | 62 | 21 | 5 |
| rotterdam1 | 38 | 79 | 117 | 0 | 5 |
| rotterdam2 | 14 | 18 | 32 | 0 | 1 |
| rotterdam3 | 19 | 30 | 49 | 0 | 1 |
| southampton | 59 | 7 | 66 | 58 | 0 |
| suez1 | 5 | 0 | 5 | 0 | 0 |
| suez2 | 5 | 0 | 5 | 0 | 0 |
| suez3 | 5 | 0 | 5 | 0 | 0 |
| suez4 | 8 | 0 | 8 | 0 | 0 |
| suez5 | 15 | 0 | 15 | 0 | 0 |
| suez6 | 31 | 4 | 35 | 19 | 4 |
| toulon | 210 | 52 | 262 | 41 | 1 |
| **TOTAL** | **716** | **267** | **983** | **231** | **29** |

## Manifest Schema

The manifest produced by `scripts/build_s2ships_manifest.py` is stored as both parquet (canonical) and CSV (human-readable) at `/data/praxis/processed/s2ships_manifest.{parquet,csv}`.

Columns:

| Column | Type | Description |
|---|---|---|
| scene_id | str | Unique scene identifier (e.g., 'brest1') |
| location | str | Location name with trailing digits stripped (e.g., 'brest') |
| is_chokepoint | bool | True for Suez and Panama scenes |
| acquisition_date | str | YYYY-MM-DD from band filename |
| coco_image_filename | str | The PNG file referenced in COCO annotations |
| coco_annotation_records | int | Number of COCO annotation records for this scene |
| n_ship | int | Polygon count, ship category |
| n_moored_ship | int | Polygon count, moored ship category |
| n_marina | int | Polygon count, marina category (excluded) |
| n_doubt | int | Polygon count, doubt category (excluded) |
| n_positive_vessels | int | n_ship + n_moored_ship |
| n_total_polygons | int | All polygons across all 4 categories |
| path_B04 | str | Absolute path to B04 (Red, 10m) TIFF |
| path_B05 | str | Absolute path to B05 (Red-edge, resampled to 10m) TIFF |
| path_B11 | str | Absolute path to B11 (SWIR, resampled to 10m) TIFF |
| path_water_mask | str | Absolute path to water mask TIFF |
| path_label_mask_png | str | Absolute path to segmentation label PNG |
| path_npy_mask | str | Absolute path to NumPy mask |
| path_rgb_preview | str | Absolute path to RGB PNG preview |
| image_height | int | 938 (uniform across dataset) |
| image_width | int | 1783 (uniform across dataset) |
| crs | str | 'EPSG:3857' (uniform across dataset) |
