# Data Acquisition Log

## S2-SHIPS Benchmark Dataset

**Date acquired:** May 24, 2026
**Workstation:** Astraea (i9-12900K, 128GB RAM, RTX 3090 24GB)
**Conda environment:** pacific (Python 3.11)
**Storage location:** `/data/praxis/raw/s2ships/`

### Source

**Primary publication:**
Ciocarlan, A., & Stoian, A. (2021). Ship Detection in Sentinel-2 Multi-Spectral Images with Self-Supervised Learning. *Remote Sensing*, 13(21), 4255. https://doi.org/10.3390/rs13214255

**Dataset citation:**
S2-SHIPS, EOTDL repository, dataset ID 6467305859028dfdbee33362. Funded by the European Space Agency under the EO Science for Society program. License: MIT.

### Acquisition Method

S2-SHIPS was acquired via the European Space Agency Earth Observation Training Data Lab (EOTDL). The dataset is also referenced via Ciocarlan's GitHub repository (`alina2204/contrastive_SSL_ship_detection`) which directs interested users to email the author for access, but the dataset is also openly redistributed through EOTDL under an MIT license uploaded by Ciocarlan herself in October 2023.

The EOTDL acquisition path was chosen because it avoided the email-and-wait friction of the GitHub access process. A courtesy email to Dr. Ciocarlan acknowledging use of the dataset for praxis research was drafted separately for transmission.

### Step-by-step procedure

```bash
# 1. Activate conda environment
conda activate pacific

# 2. Verify environment paths
which python   # ~/miniforge3/envs/pacific/bin/python
which pip      # ~/miniforge3/envs/pacific/bin/pip

# 3. Install EOTDL CLI
pip install eotdl
# Note: eotdl 2025.4.2 has a packaging bug — stac_geoparquet is imported
# but not declared as a dependency. Manual install required:
pip install stac_geoparquet

# 4. Verify CLI works
eotdl --help

# 5. Authenticate (used pre-existing EOTDL API key)
eotdl auth login
# Output: "Using API Key / Saved credentials to: ~/.cache/eotdl/creds.json
#          You are logged in as rajesh.persad@gwu.edu"

# 6. Secure credentials file
chmod 600 ~/.cache/eotdl/creds.json

# 7. Verify the dataset is visible
eotdl datasets list 2>&1 | grep -i ship
# Confirmed S2-SHIPS, Sentinel-2-Ships, and several other vessel datasets

# 8. Create destination directory
mkdir -p /data/praxis/raw/s2ships

# 9. Download the dataset (5.8 GB zipfile)
eotdl datasets get S2-SHIPS --path /data/praxis/raw/s2ships --assets --verbose
# Output: "Staging assets: 100%, 1/1 [02:51<00:00]"
#         "Data available at /data/praxis/raw/s2ships/S2-SHIPS"

# 10. Verify integrity via SHA-1 checksum
sha1sum /data/praxis/raw/s2ships/S2-SHIPS/S2-SHIPS.zip
# Expected: 53d83f86e8fcfe7b3d363228d343e3cd117647ef
# (per EOTDL catalog metadata)

# 11. Extract
cd /data/praxis/raw/s2ships
unzip -q S2-SHIPS/S2-SHIPS.zip -d extracted/
# Result: extracted/S2SHIPS/ with 16 scenes, 740MB TIFFs + 2.6GB NumPy + masks
```

### Verification results

After extraction, the following were confirmed:

- **16 scenes** across 9 locations: Brest, Toulon, Marseille, Rotterdam (×3), Southampton, Portsmouth, Rome, Suez (×6), Panama
- **12 spectral bands per scene** plus NDWI and RGB previews in `dataset_tif/`
- **NumPy mask versions** in `dataset_npy/`
- **Segmentation label masks (PNG)** in `s2ships_labels_mask/`
- **Water masks (TIFF)** in `water_mask/`
- **Two COCO annotation files** at root level: `coco-s2ships.json` (143 annotations, 3 categories) and `coco-1622202804.029818.json` (159 annotations, 4 categories)
- **Pretrained backbones** also included but not used in praxis

### Data product characterization

Inspected with rasterio (see `scripts/build_s2ships_manifest.py` for the inspection code):

| Property | Value | Notes |
|---|---|---|
| Image dimensions | 938 × 1783 pixels | All scenes identical |
| Resolution | ~10 m × 10 m | All bands resampled to 10m grid |
| CRS | EPSG:3857 (Web Mercator) | Reprojected from native UTM |
| Data type | uint16 | 16-bit per band |
| Product level | Sentinel-2 L2A (Raw) | Bottom-of-atmosphere, not L1C |
| Value range | 0 to 65535 | Saturation observed in some scenes |
| Acquisition window | April 2021 | All scenes from one month |

### Findings requiring methodological acknowledgment

The S2-SHIPS data as distributed differs from the assumptions in the praxis methodology in three ways requiring explicit documentation:

1. **Product level: L2A, not L1C.** Chapter 3 v2 currently specifies L1C top-of-atmosphere reflectance. S2-SHIPS distributes L2A bottom-of-atmosphere reflectance with the SEN2COR atmospheric correction already applied. This affects the noise model derivation since L2A has had atmospheric path radiance removed.

2. **Native 20m bands resampled to 10m.** B05 (red-edge) and B11 (SWIR) are native 20m bands at acquisition. The S2-SHIPS distribution has resampled them to a common 10m grid. The resampling method is not documented but appears to be either nearest-neighbor or bilinear based on histogram inspection.

3. **Reprojection to EPSG:3857.** Native Sentinel-2 products are delivered in the UTM zone corresponding to the tile location. S2-SHIPS has reprojected all imagery to EPSG:3857 (Web Mercator), a display projection rather than a measurement projection. Distortion increases with latitude.

These are recorded in `docs/methodological_decisions.md` with adjudication needed from the praxis advisor.

### Backup acquisition path

If EOTDL becomes unavailable or the dataset is withdrawn, the S2-SHIPS data is also available via direct request to Dr. Alina Ciocarlan (alina.ciocarlan@imt-atlantique.fr or successor address) per the GitHub repository instructions at https://github.com/alina2204/contrastive_SSL_ship_detection.
