"""
S2-SHIPS COCO Annotation Explosion
====================================

The S2-SHIPS COCO file groups multiple per-vessel polygons into single
annotation records, with the `bbox` field representing the bounding rectangle
of ALL polygons combined (isbbox: false). To use this for vessel detection
training and evaluation, each polygon must be extracted as its own annotation
with its own bounding box.

This script:
  1. Reads coco-1622202804.029818.json (the canonical 4-category file)
  2. For each annotation record, extracts each segmentation polygon as a
     separate per-vessel annotation
  3. Computes the axis-aligned bounding box of each polygon
  4. Filters to ship + moored ship categories (per methodological decision D3)
  5. Applies the 3x3 pixel minimum threshold (per Chapter 3.2.3)
  6. Writes a new COCO file with per-vessel annotations and isbbox: true
  7. Reports counts before and after each filter

Output: /data/praxis/processed/s2ships_exploded_coco.json

Coordinate space: the input COCO annotations reference the RGB PNG previews
(938 x 1783 pixels). The output preserves these PNG-space coordinates because
the multispectral TIFFs are also 938 x 1783, so the mapping is 1:1.
This 1:1 assumption is documented and should be verified visually for at least
one scene before downstream use.

Categories retained in output:
  1 -> ship
  4 -> moored ship (renumbered to 2 for cleaner downstream class indexing)

Both categories are mapped to a single "vessel" class (category_id=1) in the
output, consistent with the praxis goal of detection rather than classification.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
INPUT_COCO = Path('/data/praxis/raw/s2ships/extracted/S2SHIPS/coco-1622202804.029818.json')
OUTPUT_COCO = Path('/data/praxis/processed/s2ships_exploded_coco.json')
OUTPUT_COCO.parent.mkdir(parents=True, exist_ok=True)

# Categories to retain (by ID from the input COCO file)
POSITIVE_CATEGORY_IDS = {1, 4}  # ship=1, moored ship=4

# Minimum bounding box dimensions (pixels at B4 native 10m resolution)
MIN_BBOX_WIDTH = 3
MIN_BBOX_HEIGHT = 3


# ----------------------------------------------------------------------------
# Polygon-to-bbox conversion
# ----------------------------------------------------------------------------
def polygon_to_bbox(polygon):
    """
    Convert a COCO polygon (flat list [x1,y1,x2,y2,...]) to an axis-aligned
    bounding box in COCO format: [x_min, y_min, width, height].

    Returns None if polygon has fewer than 3 vertices.
    """
    if len(polygon) < 6:  # need at least 3 (x,y) pairs
        return None

    xs = polygon[0::2]
    ys = polygon[1::2]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    width = x_max - x_min
    height = y_max - y_min

    return [round(x_min, 2), round(y_min, 2), round(width, 2), round(height, 2)]


def bbox_passes_size_filter(bbox, min_w, min_h):
    """Return True if the bounding box meets the minimum size threshold."""
    if bbox is None:
        return False
    return bbox[2] >= min_w and bbox[3] >= min_h


# ----------------------------------------------------------------------------
# Load input
# ----------------------------------------------------------------------------
print(f'Reading {INPUT_COCO} ...')
with open(INPUT_COCO) as f:
    coco_in = json.load(f)

print(f'  {len(coco_in["images"])} images')
print(f'  {len(coco_in["annotations"])} annotation records (grouped)')
print(f'  {len(coco_in["categories"])} categories')

# Build helpers
images_by_id = {img['id']: img for img in coco_in['images']}
input_categories_by_id = {c['id']: c['name'] for c in coco_in['categories']}


# ----------------------------------------------------------------------------
# Explode annotations
# ----------------------------------------------------------------------------
exploded_annotations = []
new_annotation_id = 1

# Stats tracking
total_polygons_seen = 0
polygons_by_category = Counter()
polygons_kept_by_scene = defaultdict(int)
polygons_rejected_size = defaultdict(int)
polygons_rejected_category = defaultdict(int)
polygons_rejected_malformed = 0

for ann in coco_in['annotations']:
    image_id = ann['image_id']
    image_name = images_by_id[image_id]['file_name']
    scene_id = image_name.replace('_rgb.png', '')
    category_id = ann['category_id']
    category_name = input_categories_by_id[category_id]

    segmentations = ann.get('segmentation', [])
    if not isinstance(segmentations, list):
        # Skip if segmentation is in RLE form (unexpected here, but defensive)
        continue

    for polygon in segmentations:
        total_polygons_seen += 1
        polygons_by_category[category_name] += 1

        bbox = polygon_to_bbox(polygon)
        if bbox is None:
            polygons_rejected_malformed += 1
            continue

        # Filter: category
        if category_id not in POSITIVE_CATEGORY_IDS:
            polygons_rejected_category[category_name] += 1
            continue

        # Filter: size
        if not bbox_passes_size_filter(bbox, MIN_BBOX_WIDTH, MIN_BBOX_HEIGHT):
            polygons_rejected_size[scene_id] += 1
            continue

        # Compute polygon area (rough — using the bounding box as a proxy is
        # not the same as the true polygon area; here we approximate via
        # the bbox area since the original COCO area was for the group)
        bbox_area = bbox[2] * bbox[3]

        exploded_annotations.append({
            'id': new_annotation_id,
            'image_id': image_id,
            # Map to a single 'vessel' class (category_id=1) for detection
            'category_id': 1,
            'segmentation': [polygon],
            'bbox': bbox,
            'area': bbox_area,
            'iscrowd': 0,
            'isbbox': True,
            # Preserve the original category for traceability
            'original_category_id': category_id,
            'original_category_name': category_name,
            'source_annotation_id': ann['id'],
        })
        new_annotation_id += 1
        polygons_kept_by_scene[scene_id] += 1


# ----------------------------------------------------------------------------
# Build output COCO file
# ----------------------------------------------------------------------------
output_categories = [
    {
        'id': 1,
        'name': 'vessel',
        'supercategory': '',
        'metadata': {
            'note': 'Combined ship + moored ship from S2-SHIPS canonical COCO file',
            'source_categories': [
                {'id': 1, 'name': 'ship'},
                {'id': 4, 'name': 'moored ship'},
            ],
        },
    },
]

# Copy image metadata as-is from input
output_images = [
    {
        'id': img['id'],
        'file_name': img['file_name'],
        'width': img['width'],
        'height': img['height'],
        'scene_id': img['file_name'].replace('_rgb.png', ''),
    }
    for img in coco_in['images']
]

coco_out = {
    'info': {
        'description': 'S2-SHIPS exploded per-vessel COCO annotations (vessel class only)',
        'version': '1.0',
        'source_file': INPUT_COCO.name,
        'source_categories_kept': list(POSITIVE_CATEGORY_IDS),
        'min_bbox_width_px': MIN_BBOX_WIDTH,
        'min_bbox_height_px': MIN_BBOX_HEIGHT,
        'coordinate_space': 'PNG preview pixel coordinates (938 x 1783)',
        'coordinate_note': 'Multispectral TIFFs are also 938 x 1783; mapping is 1:1',
    },
    'images': output_images,
    'categories': output_categories,
    'annotations': exploded_annotations,
}

# ----------------------------------------------------------------------------
# Write output
# ----------------------------------------------------------------------------
print(f'\nWriting {OUTPUT_COCO} ...')
with open(OUTPUT_COCO, 'w') as f:
    json.dump(coco_out, f, indent=2)


# ----------------------------------------------------------------------------
# Summary report
# ----------------------------------------------------------------------------
print('\n' + '=' * 70)
print('Annotation Explosion Summary')
print('=' * 70)
print(f'Total polygons seen across all annotations:    {total_polygons_seen}')
print(f'  Malformed (< 3 vertices):                    {polygons_rejected_malformed}')
print()
print('Polygons by category (input):')
for cat in ['ship', 'moored ship', 'marina', 'doubt']:
    count = polygons_by_category.get(cat, 0)
    kept_by_category = (cat in {'ship', 'moored ship'})
    marker = 'KEPT' if kept_by_category else 'EXCLUDED'
    print(f'  {cat:<15} {count:>5}  [{marker}]')
print()
print(f'Total rejected by category filter:             {sum(polygons_rejected_category.values())}')
print(f'Total rejected by size filter:                 {sum(polygons_rejected_size.values())}')
print(f'Total per-vessel annotations in output:        {len(exploded_annotations)}')
print()
print('Per-scene kept vessel counts:')
print(f'{"scene_id":<15} {"kept":>5}  {"rejected_size":>15}')
all_scenes = set(polygons_kept_by_scene.keys()) | set(polygons_rejected_size.keys())
for scene in sorted(all_scenes):
    kept = polygons_kept_by_scene.get(scene, 0)
    rejected = polygons_rejected_size.get(scene, 0)
    print(f'{scene:<15} {kept:>5}  {rejected:>15}')

total_kept = sum(polygons_kept_by_scene.values())
total_rejected = sum(polygons_rejected_size.values())
print(f'{"TOTAL":<15} {total_kept:>5}  {total_rejected:>15}')
print()
print(f'Output saved to: {OUTPUT_COCO}')
print(f'Output file size: {OUTPUT_COCO.stat().st_size:,} bytes')
