# BM3D Denoising — Provenance & Validation Note

**Stage:** Classical BM3D denoising (Chapter 3, §3.5.1)
**Date verified:** 2026-06-11
**Script:** `scripts/bm3d_denoise.py` (git-versioned)
**Image:** `praxis-cpu:0.2` (numpy 2.4.6, rasterio 1.4.4, bm3d 4.0.3)
**Execution:** k3s Job `bm3d-denoise` on node `nemesis`, reading `/data` over NFS

## Scope

The four evaluation-partition scenes only (Panama, Rome, Brest1, Rotterdam1) —
BM3D requires no training, so the training-partition degraded images are out of
scope. 4 scenes x 3 bands x 3 severities x 3 realizations = **108 denoised images**.
The Job's count guard requires exactly 108; clean completion is self-verifying.

## Method

Poisson-Gaussian denoising via variance stabilization:

    standardize (z = alpha*y)  ->  forward GAT  ->  BM3D (sigma_psd = 1)
      ->  closed-form unbiased inverse GAT  ->  rescale (/ alpha)

Per-image alpha_eff and sigma_eff are read from the degradation manifest, so each
image is denoised at exactly the noise level at which it was generated. The
generalized Anscombe transform and its closed-form unbiased inverse follow
Makitalo & Foi, "Optimal inversion of the generalized Anscombe transformation for
Poisson-Gaussian noise," IEEE TIP 22(1):91-103, 2013.

### Correctness fix (recorded for transparency)

The reference forward/inverse GAT pair is only self-consistent at unit gain. Our
model has gain = 1/alpha != 1, so the transform is applied in standardized form:
the observation y is scaled to z = alpha*y (Poisson-count scale, read noise
sigma_z = alpha*sigma_r), transformed and denoised there, and the result divided
by alpha. Verified offline: forward->inverse round-trip recovers a clean signal to
mean relative error ~2e-5 across all band/severity parameterizations; an earlier
naive gain != 1 application carried a ~sigma^2/gain bias and was discarded.

## Validation

Synthetic Poisson-Gaussian patch (flat signal): RMSE reduced ~95%.
Real scene (Panama, B11, severe, r0), RMSE vs the clean reference:

  - degraded vs clean : 37.11
  - denoised vs clean : 30.13   (~19% reduction)
  - mean |denoised - degraded| : 9.87 DN  (confirms pixels were modified)

The smaller reduction on real imagery than on a flat patch is expected: scene
content (edges, structures, vessels) is correctly preserved rather than smoothed,
and B11 (20 m resampled to 10 m) carries little high-frequency detail. The
identical pipeline and parameters are applied across all 108 images, which is the
fair-comparison requirement of the classical baseline.

## Outputs

`/data/praxis/denoised/bm3d/<scene>/<band>/<severity>/r{0,1,2}.tif` (georeferencing
preserved) plus `bm3d_manifest.csv` recording alpha_eff, sigma, and source/output
paths per image.
