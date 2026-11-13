"""
S2-SHIPS PNG-to-TIFF Coordinate Verification
==============================================

Verifies the assumption that COCO annotation coordinates (in RGB PNG preview
pixel space) map 1:1 to multispectral TIFF pixel coordinates. The assumption
is used by scripts/explode_coco_annotations.py and by all downstream praxis
work that uses the exploded bounding boxes.



The verification:
  1. Selects a target scene (default: suez4 — sparse, isolated vessels)
  2. Loads the scene's RGB PNG preview and the B04 (Red) TIFF
  3. Reads the per-vessel exploded bounding boxes for this scene
  4. Renders side-by-side comparison images:
     (a) RGB PNG with all bboxes overlaid
     (b) B04 TIFF (percentile-stretched to 8-bit for visibility) with same bboxes
     (c) Difference visualization: bbox center positions on both
  5. Writes the comparison image to /data/praxis/processed/verification/

A visual inspection of the output confirms or refutes the 1:1 mapping
assumption. If bboxes sit on visible vessel signatures in BOTH images at
the same locations, the mapping is verified. If the bboxes are offset
or rotated between the two, the assumption is wrong and the exploded
COCO file needs a coordinate transform before downstream use.

Output: /data/praxis/processed/verification/{scene}_bbox_overlay.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MANIFEST_CSV = Path('/data/praxis/processed/s2ships_manifest.csv')
EXPLODED_COCO = Path('/data/praxis/processed/s2ships_exploded_coco.json')
OUTPUT_DIR = Path('/data/praxis/processed/verification')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default verification scene: suez4 is sparse (8 vessels) and isolated
# Override via command-line: python verify_png_tiff_coords.py panama
DEFAULT_SCENE = 'suez4'
TARGET_SCENE = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENE


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def percentile_stretch_to_8bit(arr, low_pct=2, high_pct=98, mask_saturation=True):
    """
    Stretch a uint16 array to 8-bit using percentile clipping for display.

    Saturated pixels (65535) are optionally masked from the percentile
    calculation so they don't dominate the high end.
    """
    if mask_saturation:
        valid = arr[arr < 65535]
    else:
        valid = arr.ravel()

    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    lo = np.percentile(valid, low_pct)
    hi = np.percentile(valid, high_pct)
    if hi <= lo:
        hi = lo + 1
    stretched = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (stretched * 255).astype(np.uint8)


def grayscale_to_rgb(arr):
    """Convert uint8 grayscale to RGB by stacking channels."""
    return np.stack([arr, arr, arr], axis=-1)


def draw_bboxes(img_array, bboxes, color=(255, 0, 0), width=2, labels=None):
    """
    Draw bounding boxes on an RGB image array.

    bboxes: list of [x, y, w, h] in COCO format
    img_array: uint8 RGB numpy array
    Returns: PIL Image with boxes drawn
    """
    img = Image.fromarray(img_array.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        # Inflate slightly so small bboxes are visible
        x0, y0 = x, y
        x1, y1 = x + w, y + h
        for offset in range(width):
            draw.rectangle(
                [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                outline=color,
            )
        if labels and i < len(labels):
            draw.text((x1 + 3, y0), str(labels[i]), fill=color)
    return img


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
print(f'Verifying scene: {TARGET_SCENE}')
print(f'Loading manifest from {MANIFEST_CSV} ...')
import pandas as pd
manifest = pd.read_csv(MANIFEST_CSV)
scene_row = manifest[manifest['scene_id'] == TARGET_SCENE]
if scene_row.empty:
    raise SystemExit(f'Scene {TARGET_SCENE} not found in manifest.')
scene_row = scene_row.iloc[0]

rgb_png_path = scene_row['path_rgb_preview']
b04_tif_path = scene_row['path_B04']
print(f'  RGB preview: {rgb_png_path}')
print(f'  B04 TIFF:    {b04_tif_path}')

# Load the exploded COCO
print(f'Loading exploded annotations from {EXPLODED_COCO} ...')
with open(EXPLODED_COCO) as f:
    coco = json.load(f)
scene_image = next(
    (img for img in coco['images'] if img.get('scene_id') == TARGET_SCENE),
    None,
)
if scene_image is None:
    raise SystemExit(f'No scene_id={TARGET_SCENE} found in exploded COCO images.')
scene_image_id = scene_image['id']
scene_bboxes = [
    ann['bbox']
    for ann in coco['annotations']
    if ann['image_id'] == scene_image_id
]
print(f'  Vessels in {TARGET_SCENE}: {len(scene_bboxes)}')


# ----------------------------------------------------------------------------
# Load images
# ----------------------------------------------------------------------------
print('Loading RGB PNG ...')
rgb_img = np.array(Image.open(rgb_png_path).convert('RGB'))
print(f'  PNG shape: {rgb_img.shape}  dtype: {rgb_img.dtype}')

print('Loading B04 TIFF ...')
with rasterio.open(b04_tif_path) as src:
    b04_raw = src.read(1)
print(f'  TIFF shape: {b04_raw.shape}  dtype: {b04_raw.dtype}  '
      f'min={b04_raw.min()}  max={b04_raw.max()}')

# Stretch B04 to 8-bit for visualization
b04_stretched = percentile_stretch_to_8bit(b04_raw)
b04_rgb = grayscale_to_rgb(b04_stretched)
print(f'  After stretch: min={b04_stretched.min()}  max={b04_stretched.max()}')


# ----------------------------------------------------------------------------
# Sanity check: dimensions must match
# ----------------------------------------------------------------------------
png_h, png_w = rgb_img.shape[:2]
tif_h, tif_w = b04_raw.shape
print(f'\nDimension check:')
print(f'  PNG: {png_w} x {png_h}')
print(f'  TIFF: {tif_w} x {tif_h}')
if (png_w, png_h) != (tif_w, tif_h):
    print(f'  WARNING: dimensions differ! 1:1 mapping cannot hold.')
else:
    print(f'  OK: dimensions match.')


# ----------------------------------------------------------------------------
# Render overlays
# ----------------------------------------------------------------------------
print(f'\nRendering bbox overlays for {len(scene_bboxes)} vessels ...')
labels = [str(i + 1) for i in range(len(scene_bboxes))]
rgb_overlay = draw_bboxes(rgb_img, scene_bboxes, color=(255, 0, 0), width=2, labels=labels)
b04_overlay = draw_bboxes(b04_rgb, scene_bboxes, color=(255, 0, 0), width=2, labels=labels)


# ----------------------------------------------------------------------------
# Compose side-by-side comparison
# ----------------------------------------------------------------------------
print('Composing side-by-side image ...')
gap = 20
total_w = png_w + tif_w + gap
total_h = max(png_h, tif_h) + 40  # extra room for title text
comparison = Image.new('RGB', (total_w, total_h), color=(20, 20, 20))
comparison.paste(rgb_overlay, (0, 40))
comparison.paste(b04_overlay, (png_w + gap, 40))

# Title
draw = ImageDraw.Draw(comparison)
draw.text((10, 10), f'{TARGET_SCENE}: RGB PNG (left)  |  B04 TIFF stretched (right)',
          fill=(255, 255, 255))
draw.text((10, 25), f'{len(scene_bboxes)} vessels overlaid in red — confirm same positions on both',
          fill=(200, 200, 200))


# ----------------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------------
out_compare = OUTPUT_DIR / f'{TARGET_SCENE}_bbox_overlay_comparison.png'
out_rgb = OUTPUT_DIR / f'{TARGET_SCENE}_bbox_on_rgb.png'
out_b04 = OUTPUT_DIR / f'{TARGET_SCENE}_bbox_on_b04.png'

comparison.save(out_compare)
rgb_overlay.save(out_rgb)
b04_overlay.save(out_b04)

print(f'\nOutputs written to:')
print(f'  Comparison:  {out_compare}')
print(f'  RGB only:    {out_rgb}')
print(f'  B04 only:    {out_b04}')
print()
print('VERIFICATION INSTRUCTIONS:')
print('  Open the comparison image and visually inspect.')
print('  Pass criterion: each red bbox sits on a visible vessel signature')
print('                   at the SAME position in both the RGB and B04 views.')
print('  If any bbox is offset between RGB and B04, the 1:1 mapping assumption')
print('  is violated and the explosion pipeline needs a coordinate transform.')

