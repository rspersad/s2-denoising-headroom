"""
S2-SHIPS Data Manifest Builder
================================

Walks the extracted S2-SHIPS directory and produces a structured catalog
of all 16 scenes, linking imagery, masks, annotations, and metadata.

Output: /data/praxis/processed/s2ships_manifest.parquet
        /data/praxis/processed/s2ships_manifest.csv  (human-readable backup)

Run from anywhere; paths are absolute.
"""

import json
import os
import re
import glob
from collections import defaultdict
from pathlib import Path

import pandas as pd
import rasterio

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
S2SHIPS_ROOT = Path('/data/praxis/raw/s2ships/extracted/S2SHIPS')
COCO_FILE = S2SHIPS_ROOT / 'coco-1622202804.029818.json'  # 4-category, 159 annotations
OUTPUT_DIR = Path('/data/praxis/processed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Categories we treat as positive vessel annotations
# (advisor decision — flagged in change log; current choice: ship + moored ship)
POSITIVE_CATEGORIES = {'ship', 'moored ship'}
EXCLUDED_CATEGORIES = {'marina', 'doubt'}

# Bands we care about for the praxis (per Chapter 3 v2)
PRAXIS_BANDS = ['B04', 'B05', 'B11']

# Chokepoint locations called out in Chapter 1
CHOKEPOINT_SCENES = {
    'suez1', 'suez2', 'suez3', 'suez4', 'suez5', 'suez6',  # Suez Canal
    'panama',                                                # Panama Canal entrance
}

# ----------------------------------------------------------------------------
# Load COCO annotations
# ----------------------------------------------------------------------------
with open(COCO_FILE) as f:
    coco = json.load(f)

categories_by_id = {c['id']: c['name'] for c in coco['categories']}
images_by_id = {img['id']: img for img in coco['images']}

# Build per-image annotation stats
ann_stats = defaultdict(lambda: defaultdict(int))  # image_filename -> category -> polygon_count
ann_records = defaultdict(int)                      # image_filename -> annotation_record_count

for ann in coco['annotations']:
    img_meta = images_by_id[ann['image_id']]
    img_name = img_meta['file_name']
    cat_name = categories_by_id[ann['category_id']]
    num_polys = len(ann.get('segmentation', []))
    ann_stats[img_name][cat_name] += num_polys
    ann_records[img_name] += 1


# ----------------------------------------------------------------------------
# Scene discovery
# ----------------------------------------------------------------------------
def scene_id_from_imagename(filename):
    """
    Map COCO image file_name (e.g., 'brest1_rgb.png') to scene_id ('brest1').
    """
    return filename.replace('_rgb.png', '').replace('.png', '')


def find_band_tiff(scene_dir, band_code):
    """
    Locate the TIFF file for a given band in a scene directory.
    Files are named like:
      2021-04-14-00_00_2021-04-14-23_59_Sentinel-2_L2A_B04_(Raw).tiff
      2021-04-14-00_00_2021-04-14-23_59_Sentinel-2_L2A_B04_Raw.tiff
    """
    candidates = glob.glob(str(scene_dir / f'*_{band_code}_*.tiff'))
    candidates += glob.glob(str(scene_dir / f'*_{band_code}.tiff'))
    candidates = [c for c in candidates if 'True_color' not in c and 'NDWI' not in c]
    if not candidates:
        return None
    return candidates[0]


def find_water_mask(scene_id):
    """Locate the water mask for a scene."""
    candidates = [
        S2SHIPS_ROOT / 'water_mask' / f'{scene_id}_water.tif',
        S2SHIPS_ROOT / 'dataset_tif' / scene_id / f'{scene_id}_water.tif',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def find_label_mask(scene_id):
    """Locate the segmentation label mask for a scene (PNG)."""
    pattern = str(S2SHIPS_ROOT / 's2ships_labels_mask' / f'*_mask_{scene_id}_rgb.png')
    candidates = glob.glob(pattern)
    return candidates[0] if candidates else None


def find_npy_mask(scene_id):
    """Locate the NumPy mask for a scene."""
    pattern = str(S2SHIPS_ROOT / 'dataset_npy' / f'*_mask_{scene_id}.npy')
    candidates = glob.glob(pattern)
    return candidates[0] if candidates else None


def parse_acquisition_date(scene_dir):
    """
    Extract acquisition date from a band TIFF filename.
    Filename format: 2021-04-14-00_00_2021-04-14-23_59_Sentinel-2_L2A_...
    """
    candidates = glob.glob(str(scene_dir / '*Sentinel-2_L2A*.tiff'))
    if not candidates:
        return None
    basename = os.path.basename(candidates[0])
    match = re.match(r'(\d{4}-\d{2}-\d{2})', basename)
    return match.group(1) if match else None


def location_from_scene_id(scene_id):
    """
    Extract location name from scene_id (strip trailing digits).
    'rotterdam1' -> 'rotterdam'
    'suez6' -> 'suez'
    'brest1' -> 'brest'
    """
    return re.sub(r'\d+$', '', scene_id)


# ----------------------------------------------------------------------------
# Build the manifest
# ----------------------------------------------------------------------------
records = []

dataset_tif_dir = S2SHIPS_ROOT / 'dataset_tif'
scene_dirs = sorted([d for d in dataset_tif_dir.iterdir() if d.is_dir()])

for scene_dir in scene_dirs:
    scene_id = scene_dir.name
    location = location_from_scene_id(scene_id)
    img_name = f'{scene_id}_rgb.png'

    # Find band TIFFs
    band_paths = {}
    band_metadata = {}
    for band in PRAXIS_BANDS:
        path = find_band_tiff(scene_dir, band)
        band_paths[band] = path
        if path:
            try:
                with rasterio.open(path) as src:
                    band_metadata[band] = {
                        'shape': src.shape,
                        'crs': str(src.crs),
                        'transform': tuple(src.transform)[:6],
                    }
            except Exception as e:
                band_metadata[band] = {'error': str(e)}

    # Get annotation stats
    cat_counts = ann_stats.get(img_name, {})
    ship_count = cat_counts.get('ship', 0)
    moored_count = cat_counts.get('moored ship', 0)
    marina_count = cat_counts.get('marina', 0)
    doubt_count = cat_counts.get('doubt', 0)

    record = {
        'scene_id': scene_id,
        'location': location,
        'is_chokepoint': scene_id in CHOKEPOINT_SCENES,
        'acquisition_date': parse_acquisition_date(scene_dir),
        'coco_image_filename': img_name,
        'coco_annotation_records': ann_records.get(img_name, 0),

        # Polygon counts by category
        'n_ship': ship_count,
        'n_moored_ship': moored_count,
        'n_marina': marina_count,
        'n_doubt': doubt_count,
        'n_positive_vessels': ship_count + moored_count,
        'n_total_polygons': ship_count + moored_count + marina_count + doubt_count,

        # File paths
        'path_B04': band_paths.get('B04'),
        'path_B05': band_paths.get('B05'),
        'path_B11': band_paths.get('B11'),
        'path_water_mask': find_water_mask(scene_id),
        'path_label_mask_png': find_label_mask(scene_id),
        'path_npy_mask': find_npy_mask(scene_id),
        'path_rgb_preview': str(scene_dir / f'{scene_id}_rgb.png') if (scene_dir / f'{scene_id}_rgb.png').exists() else None,

        # Image dimensions (assuming all bands are same shape, which we verified earlier)
        'image_height': band_metadata.get('B04', {}).get('shape', [None, None])[0] if band_metadata.get('B04') else None,
        'image_width': band_metadata.get('B04', {}).get('shape', [None, None])[1] if band_metadata.get('B04') else None,
        'crs': band_metadata.get('B04', {}).get('crs'),
    }
    records.append(record)


# ----------------------------------------------------------------------------
# Build DataFrame and save
# ----------------------------------------------------------------------------
manifest = pd.DataFrame(records)
manifest = manifest.sort_values(['is_chokepoint', 'location', 'scene_id'], ascending=[False, True, True])

# Save in two formats
parquet_path = OUTPUT_DIR / 's2ships_manifest.parquet'
csv_path = OUTPUT_DIR / 's2ships_manifest.csv'
manifest.to_parquet(parquet_path, index=False)
manifest.to_csv(csv_path, index=False)

# ----------------------------------------------------------------------------
# Summary report
# ----------------------------------------------------------------------------
print('=' * 80)
print(f'S2-SHIPS Manifest Built')
print('=' * 80)
print(f'Output Parquet: {parquet_path}')
print(f'Output CSV:     {csv_path}')
print(f'\nTotal scenes: {len(manifest)}')
print(f'Locations: {manifest["location"].nunique()}')
print(f'Chokepoint scenes: {manifest["is_chokepoint"].sum()}')
print(f'\nPositive vessels (ship + moored): {manifest["n_positive_vessels"].sum()}')
print(f'Excluded marina annotations:       {manifest["n_marina"].sum()}')
print(f'Excluded doubt annotations:        {manifest["n_doubt"].sum()}')
print(f'\nPer-scene summary:')
print(manifest[['scene_id', 'location', 'is_chokepoint', 'acquisition_date',
                'n_ship', 'n_moored_ship', 'n_marina', 'n_doubt',
                'n_positive_vessels']].to_string(index=False))

# Check for any missing critical files
print(f'\n--- Integrity checks ---')
missing_b04 = manifest[manifest['path_B04'].isna()]
missing_b05 = manifest[manifest['path_B05'].isna()]
missing_b11 = manifest[manifest['path_B11'].isna()]
missing_water = manifest[manifest['path_water_mask'].isna()]
missing_label = manifest[manifest['path_label_mask_png'].isna()]
missing_npy = manifest[manifest['path_npy_mask'].isna()]

print(f'Scenes missing B04: {len(missing_b04)}  {missing_b04["scene_id"].tolist() if len(missing_b04) else ""}')
print(f'Scenes missing B05: {len(missing_b05)}  {missing_b05["scene_id"].tolist() if len(missing_b05) else ""}')
print(f'Scenes missing B11: {len(missing_b11)}  {missing_b11["scene_id"].tolist() if len(missing_b11) else ""}')
print(f'Scenes missing water mask: {len(missing_water)}  {missing_water["scene_id"].tolist() if len(missing_water) else ""}')
print(f'Scenes missing label mask: {len(missing_label)}  {missing_label["scene_id"].tolist() if len(missing_label) else ""}')
print(f'Scenes missing NPY mask:   {len(missing_npy)}  {missing_npy["scene_id"].tolist() if len(missing_npy) else ""}')
