# Third-party notices

This repository redistributes or derives from the following works. Each notice is required by the respective license and must be retained in any redistribution.

## Copernicus Sentinel-2 data
Contains modified Copernicus Sentinel data [acquisition year(s) of the scenes — fill from docs/DATA_CARD.md].
Licence: Copernicus Sentinel data legal notice (free, full and open access; reproduction, distribution and adaptation permitted with attribution).
https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice

## S2-SHIPS vessel annotations
Ciocarlan, A., & Stoian, A. (2021). Ship Detection in Sentinel 2 Multi-Spectral Images with Self-Supervised Learning. Remote Sensing, 13(21), 4255. https://doi.org/10.3390/rs13214255
Dataset licence: MIT (EOTDL catalogue entry; source repository https://github.com/alina2204/contrastive_SSL_ship_detection). Licence text: LICENSES/S2-SHIPS-MIT.txt. Annotation-derived files redistributed under that licence with this notice retained.

## YOLOv9 and Ultralytics
Wang, C.-Y., Yeh, I.-H., & Liao, H.-Y. M. (2024). YOLOv9: Learning what you want to learn using programmable gradient information. ECCV 2024, LNCS 15089. Original implementation: GPL-3.0.
Ultralytics YOLO (training and evaluation framework used for all detectors in this work): AGPL-3.0, https://github.com/ultralytics/ultralytics
The trained detector weights published with this work are released under AGPL-3.0.

## Denoising methods (run from their published implementations; not vendored)
- BM3D — Dabov, Foi, Katkovnik & Egiazarian (2007); Python package `bm3d` (Tampere University), academic/non-commercial terms.
- DnCNN — Zhang, Zuo, Chen, Meng & Zhang (2017).
- Restormer — Zamir et al. (2022); code and weights under an academic, non-commercial license. No Restormer weights are redistributed here.
- Noise2Void — Krull, Buchholz & Jug (2019); BSD-3-Clause.
- Blind2Unblind — Wang et al. (2022).
Consult each project's license before commercial use of any pipeline built from this code.

## Python dependencies
Listed in code/requirements.txt with their licenses in docs/DEPENDENCY_LICENSES.md (generated with `pip-licenses`).
