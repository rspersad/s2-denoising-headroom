# Large files: archive record

The tiles, degraded variants, label files and detector weights are held in a read-only archive record and are not in this Git repository.

| Item | File in archive | Size | SHA-256 |
|---|---|---|---|
| Native tiles, clean (B04, B05, B11) | `tiles_clean.tar.zst` | 87 MB | see MANIFEST.sha256 |
| Degraded tiles, nominal / moderate / severe × 3 draws | `tiles_degraded.tar.zst` | 1.09 GB | see MANIFEST.sha256 |
| Denoised tiles, five methods, and the severity-matched arm | `tiles_denoised.tar.zst` | 1.59 GB | see MANIFEST.sha256 |
| YOLO label files (annotation-derived; MIT, see docs/PERMISSIONS.md) | `labels_yolo.tar.zst` | 22 KB | see MANIFEST.sha256 |
| YOLOv9 detectors, m/c/e × seeds {s1,s2,s3} | `detectors_yolov9.tar.zst` | 597 MB | see MANIFEST.sha256 |

Archive: **10.5281/zenodo.24209175 — https://doi.org/10.5281/zenodo.24209175** — Zenodo record, version 1.0.0 (immutable; later corrections are new versions).

`tools/fetch_archive.py` downloads these files into `data/archive/` and verifies them against `MANIFEST.sha256`.

Copernicus notice for all imagery: *Contains modified Copernicus Sentinel data [acquisition year]*.
