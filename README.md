# Perceptual Quality Versus Detection Performance — Data and Code

Companion repository for the Doctor of Engineering Praxis

> Persad, R. S. (2026). *Perceptual Quality Versus Detection Performance: Recoverable Headroom as a Basis for Denoising Decisions in Sentinel-2 Vessel Detection.* The George Washington University, School of Engineering and Applied Science.

This repository is published so that the measurements reported in the Praxis can be inspected, re-analyzed and reproduced. It is **read-only for the public**: anyone may download and use the contents under the licenses below; only the author can change what is published here, and the published releases are immutable (see *Integrity*).

## What is here

| Path | Contents | License |
|---|---|---|
| `code/` | The `s2_analysis` scripts: tiling and dataset construction, noise calibration and degradation, denoiser execution, detector training and evaluation, statistics and figures | AGPL-3.0 (`LICENSE`) |
| `results/` | The primary results table (660 rows), per-block perceptual and detection scores, the ceiling-contrast table, calibration values (Table 3.1 of the Praxis) and the statistical outputs behind every table in Chapter 4 | CC BY 4.0 (`LICENSE-DATA`) |
| `configs/` | Degradation parameters, detector training configurations, random seeds, split definitions | CC BY 4.0 |
| `docs/` | Data card, degradation provenance, progress log excerpts, variable dictionary for `results/` | CC BY 4.0 |
| `MANIFEST.sha256` | SHA-256 checksum of every published file, including those held in the external archive | — |

Large binary artifacts are **not** stored in the Git history (GitHub limits files to 100 MB). They are held in a versioned, read-only archive and listed in the manifest:

| Artifact | Where | License |
|---|---|---|
| 640 × 640 native-resolution tiles (B04, B05, B11), degraded variants at three severities, YOLO label files | Archive record — see `data/README.md` | See *Data provenance* |
| Nine trained YOLOv9 detectors (m / c / e × three seeds) | Archive record — see `data/README.md` | AGPL-3.0 |

## Reproducing the analysis

```bash
git clone https://github.com/rspersad/s2-denoising-headroom.git
cd s2-denoising-headroom
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
python tools/fetch_archive.py          # downloads the archive files listed in data/README.md
python tools/verify_manifest.py        # confirms every file matches MANIFEST.sha256
```

The statistical tables of Chapter 4 are regenerated from `results/primary_table.csv` alone; the tiles and detectors are needed only to re-run the factorial itself. `docs/REPRODUCING.md` gives the full order of operations and the commit of the private working repository that each published file was taken from.

## Data provenance and attribution

* **Sentinel-2 imagery.** All image tiles are derived from Copernicus Sentinel-2 Level-2A products. They are redistributed under the [Copernicus Sentinel data licence](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice), which permits reproduction, distribution and adaptation with attribution. Required notice: **"Contains modified Copernicus Sentinel data [acquisition year]"**.
* **Vessel annotations.** The vessel annotations originate from the S2-SHIPS dataset (Ciocarlan & Stoian, 2021, *Remote Sensing* 13(21), 4255, https://doi.org/10.3390/rs13214255), obtained from its authors. S2-SHIPS is distributed on request and carries no public license; the annotation-derived files in this repository are published with the written permission of the S2-SHIPS authors (see `docs/PERMISSIONS.md`). Users of those files must cite Ciocarlan & Stoian (2021) in addition to this repository.
* **Detectors.** The YOLOv9 models were trained and evaluated with the Ultralytics implementation (AGPL-3.0) of YOLOv9 (Wang, Yeh & Liao, 2024). Weights are released under AGPL-3.0 accordingly.
* **Denoisers.** BM3D, DnCNN, Restormer, Noise2Void and Blind2Unblind were run from their published implementations under their own licenses; none is vendored here. Note that the Restormer code and pretrained weights are released by their authors for academic, non-commercial use only; any fine-tuned Restormer weights are therefore **not** published in this repository.

## How to cite

```bibtex
@phdthesis{persad2026headroom,
  author = {Persad, Rajesh Surendra},
  title  = {Perceptual Quality Versus Detection Performance: Recoverable Headroom as a Basis for Denoising Decisions in Sentinel-2 Vessel Detection},
  school = {The George Washington University},
  year   = {2026},
  type   = {Doctor of Engineering Praxis}
}
```

For the data and code specifically, cite the archived release (DOI in `CITATION.cff` once minted). Data users must also carry the Copernicus notice above and cite Ciocarlan & Stoian (2021) for annotation-derived files.

## Integrity

* Releases are tagged (`v1.0.0` corresponds to the Praxis as approved by the committee). Tags are protected and cannot be moved or deleted.
* The default branch is protected: no direct pushes, no force-pushes, no deletion; changes arrive only through reviewed pull requests from the author.
* `MANIFEST.sha256` lists every published file. `tools/verify_manifest.py` checks a local copy against it.
* The archive record holding the large files is immutable once published; corrections are issued as a new version with its own DOI, and the old version remains available.

## Contact

Rajesh Surendra Persad — open an issue in this repository for questions about the data or code.
