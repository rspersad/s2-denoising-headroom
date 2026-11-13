# Synthetic Degradation — Provenance & Reproducibility Note

**Stage:** Synthetic degradation (Chapter 3, §3.4)
**Date verified:** 2026-06-11
**Script:** `scripts/apply_degradation.py` (git-versioned)
**Inputs:** `s2ships_manifest.csv` (16 scenes), `noise_calibration.json`

## Factorial

16 scenes × 3 bands (B04/B05/B11) × 3 severities (nominal/moderate/severe)
× 3 noise realizations = **432 degraded GeoTIFFs**, recorded with per-image
effective parameters and seeds in `degradation_manifest.csv`.

## Noise model

Poisson–Gaussian in pixel-value units (reflectance × QV = 10000):

    y = (1/α_eff)·Poisson(α_eff·I) + N(0, σ_eff²)

with, per band/severity,
`α_eff = alpha_calibrated × alpha_scale` and
`σ_eff = read_noise_sigma_r × sigma_r_scale`
(nominal 1.0/1.0, moderate 0.667/2.0, severe 0.5/4.0).

Each (scene, band, severity, realization) is seeded by a SHA-256 of those four
fields, making every realization independent yet regenerable bit-for-bit,
independent of process or `PYTHONHASHSEED`.

## Cross-environment determinism result

The full set was generated independently in two environments:

| Environment | Execution | rasterio / GDAL |
|---|---|---|
| `pacific` conda env | direct on Astraea | 1.4.4 / conda-forge |
| `praxis-cpu:0.1` image | k3s Job (pod `degrade-s2ships-p8gfk`, 39 s) | 1.4.4 / wheel 3.10.3 |

The cluster run was confirmed independent: Job `Complete 1/1`, log line
`[total] wrote 432 images (expected 432)`, and output mtimes (00:40:51)
coincident with the Job, not the earlier conda pass.

Digests over all 432 outputs (identical across both environments):

- **Pixel-level** (band arrays, environment-independent):
  `d196a5281c4fa38716134d3fe5f364ce44e8e08833703f6393249d1d81e08775`
- **File-level** (whole GeoTIFF bytes):
  `ffed5486c558b933bd62a7bca518ab5f3dd0bedcaed7c008b9801b2f5719720a`

## What this demonstrates

1. The seeded Poisson–Gaussian generator is deterministic and
   environment-independent (pixel digest matches across stacks).
2. Calibration and severity scaling apply identically in both environments.
3. The containerized cluster pipeline reproduces the reference set exactly.
4. The matching *file*-level digest further shows the conda-forge and pip-wheel
   GDAL writers serialize byte-identically (no encoder divergence).

The degraded set is therefore a deterministic, reproducible artifact: any party
with the clean inputs, `noise_calibration.json`, and `apply_degradation.py`
regenerates these 432 images bit-for-bit.
