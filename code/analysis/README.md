# Praxis: Perceptual Quality Versus Detection Performance

**An Evaluation of AI Denoising Methods for Sentinel-2 Vessel Detection**

A doctoral praxis investigating the conditions under which AI-based denoising methods improve, degrade, or have no measurable impact on vessel detection performance on Sentinel-2 imagery.

**Status:** Work in progress.

## Overview

AI-based denoising is commonly applied as a preprocessing step in operational vessel detection pipelines. These methods are typically selected and tuned according to perceptual quality metrics (PSNR, SSIM), under the implicit assumption that improvements in perceived image quality translate to improvements in downstream detection performance.

This assumption is theoretically problematic and empirically unverified in the satellite imagery domain. The perception-distortion tradeoff (Blau & Michaeli, 2018) establishes that optimizing for perceptual fidelity necessarily displaces the processed signal relative to the underlying data distribution, and prior work in natural images demonstrates that the relationship between perceptual quality and deep network performance is neither monotonic nor proportional (Dodge & Karam, 2016).

This praxis evaluates five denoising methods—BM3D, DnCNN, Restormer, Noise2Void, and Blind2Unblind—spanning classical, supervised deep learning, and self-supervised paradigms, applied to Sentinel-2 imagery under controlled synthetic degradation. Both perceptual metrics (PSNR, SSIM) and downstream vessel detection performance (mAP@0.5, mAP@0.5:0.95 via YOLOv8) are computed across three spectral bands and multiple degradation severities.

Statistical significance is evaluated using the Wilcoxon signed-rank test for pairwise comparisons and the Friedman test with Nemenyi post-hoc analysis for comparisons across multiple methods.

## Scope

This work is intentionally restricted to:

- **Sensor:** Sentinel-2 Multispectral Instrument (Levels 1C and 2A)
- **Spectral bands:** 10-m visible, 20-m red-edge, 20-m short-wave infrared
- **Task:** Vessel detection (localization and confidence-rated identification)
- **Detector:** YOLOv8
- **Denoisers:** BM3D, DnCNN, Restormer, Noise2Void, Blind2Unblind
- **Degradation:** Controlled synthetic degradation calibrated to published Sentinel-2 sensor noise specifications

This bounded scope sacrifices cross-sensor and cross-task generalization in exchange for the controlled comparison required to make statistically defensible claims about denoising effects.

## Environment

The Pacific conda environment captures all Python dependencies:

```bash
mamba env create -f pacific.yml
conda activate pacific
```

Key components: PyTorch 2.4 (CUDA 12.1), rasterio/GDAL for satellite imagery, OpenCV, statsmodels for mixed-effects analysis, ultralytics for YOLOv8.

## Repository Structure
```
src/                  Implementation code
data/               Dataset loaders, splits, degradation generation
denoisers/          Denoiser implementations
detectors/          Object detection wrappers
metrics/            Perceptual and task metrics
training/           Training loops
evaluation/         Evaluation pipelines
utils/              Shared helpers
configs/              Experiment YAML configurations
scripts/              Data preparation, sweeps, one-off scripts
notebooks/            Exploratory Jupyter notebooks
tests/                Unit tests
```
## Data

Praxis data is not stored in this repository. On the development machine (Astraea), data lives at `/data/praxis/`:
```
raw/sentinel2/{visible,red-edge,swir}/       Untouched Sentinel-2 imagery
synthetic-degradations/sentinel2/            Degraded versions (band × severity)
denoised/{method}/{band}/                    Outputs per denoiser per band
predictions/yolov8/                          Detection outputs (preserved per Chapter 3.8.1)
annotations/                                 Ground truth bounding boxes
splits/                                      Train/val/test partitions
models/                                      Trained checkpoints
```
Sentinel-2 imagery is obtainable from the Copernicus Open Access Hub.

## Reproducibility

Per Chapter 3.8.1, this praxis commits to:

- Pinned dependency versions (`pacific.yml`)
- Fixed random seeds for all stochastic processes
- Preserved predictions on test sets for re-analysis
- Documented training scripts and configurations
- Published splits files defining train/val/test partitions

## Author

Rajesh Persad

## License

This is a doctoral research project. Code license to be determined at publication.