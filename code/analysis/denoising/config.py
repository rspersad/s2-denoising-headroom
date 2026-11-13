"""
Central configuration for the learned-denoiser training pipeline.

Everything that depends on the on-disk data layout lives here so it can be
adjusted in one place. Run `python -m denoising.audit_data` FIRST on Astraea to
verify these assumptions against your actual /data layout before training.

Praxis: "Perceptual Quality Versus Detection Performance: An Evaluation of AI
Denoising Methods for Sentinel-2 Vessel Detection" (R. Persad, GWU).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  -- ADJUST IF YOUR LAYOUT DIFFERS (audit_data.py will tell you)
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("PRAXIS_DATA_ROOT", "/data/praxis"))
PROCESSED = DATA_ROOT / "processed"

# Canonical noise calibration produced by Appendix C.
NOISE_CALIBRATION_JSON = PROCESSED / "noise_calibration.json"

# The real S2-SHIPS manifest is a CSV with explicit per-band raster paths
# (columns path_B04/path_B05/path_B11) plus scene_id. Band rasters are read
# directly from those columns -- no band-index guessing.
MANIFEST_CSV = PROCESSED / "s2ships_manifest.csv"

# Trained denoiser checkpoints are written here, one subdir per method.
CHECKPOINT_ROOT = Path(os.environ.get("PRAXIS_CKPT_ROOT", str(DATA_ROOT / "models" / "denoisers")))

# Restormer pretrained checkpoint (official Gaussian-denoising weights).
# Use the GRAYSCALE blind checkpoint: our denoisers operate on single S2 bands.
RESTORMER_PRETRAINED = Path(
    os.environ.get("RESTORMER_PRETRAINED", str(DATA_ROOT / "pretrained" / "gaussian_gray_denoising_blind.pth"))
)

# ---------------------------------------------------------------------------
# PARTITION (three-way design; Chapter 3 §3.6.2 as revised)
# ---------------------------------------------------------------------------
# EVAL scenes are the factorial scenes -- held out of ALL denoiser training to
# prevent leakage. The 12 non-eval scenes train the denoisers. Within those 12
# the detector additionally uses DETECTOR_VAL for checkpoint selection, but the
# denoisers may use all 12 since none are factorial scenes.
EVAL_SCENES = ["panama", "rome", "brest1", "rotterdam1"]   # 4, factorial (held out)
DETECTOR_VAL_SCENES = ["marseille", "rotterdam2"]          # 2, YOLO checkpoint-selection
# DETECTOR_TRAIN = suez1-6, toulon, rotterdam3, southampton, portsmouth (10)

# Band -> manifest column holding that band's single-band raster path.
BAND_PATH_COL = {"B04": "path_B04", "B05": "path_B05", "B11": "path_B11"}

# ---------------------------------------------------------------------------
# Bands / severities
# ---------------------------------------------------------------------------
BANDS = ["B04", "B05", "B11"]
SEVERITIES = ["nominal", "moderate", "severe"]

# Severity scaling applied to the nominal (alpha, sigma_r). These mirror the
# Appendix C definitions; the loader prefers values baked into
# noise_calibration.json and only falls back to these factors if the JSON does
# not already enumerate per-severity parameters.
SEVERITY_FACTORS = {
    "nominal":  {"alpha_factor": 1.0,        "sigma_factor": 1.0},
    "moderate": {"alpha_factor": 2.0 / 3.0,  "sigma_factor": 2.0},  # 1/3 gain reduction, 2x read
    "severe":   {"alpha_factor": 0.5,        "sigma_factor": 4.0},  # 1/2 gain, 4x read
}

# Nominal fallback parameters (Appendix C) used only if the JSON is absent.
NOMINAL_FALLBACK = {
    "B04": {"alpha": 8.4187,  "sigma_r": 5.00},
    "B05": {"alpha": 7.9692,  "sigma_r": 5.00},
    "B11": {"alpha": 19.1154, "sigma_r": 2.00},
}

# ---------------------------------------------------------------------------
# Pixel scaling
# ---------------------------------------------------------------------------
# Noise is applied in DN units (Appendix C, Decision D12). For network input we
# divide by REFLECTANCE_SCALE so values sit roughly in [0, ~1]. audit_data.py
# reports the actual value range so you can confirm this scale.
# ---------------------------------------------------------------------------
# Radiometric harmonization
# ---------------------------------------------------------------------------
# The S2-SHIPS "(Raw)" rasters store reflectance scaled by 65535 (full 16-bit
# range), confirmed empirically via SWIR-over-water (B11 water reflectance only
# lands near 0 under /65535). Appendix C calibrated alpha/sigma_r for the
# reflectance x 10000 domain. So every raw raster is harmonized into the
# calibration domain BEFORE any noise injection or denoising:
#       calibrated_value = raw_dn * (CALIB_QV / NATIVE_QV)
# Rule: harmonize exactly ONCE, at the point a raw S2-SHIPS raster is read.
# (apply_degradation.py and data.read_band both do this; downstream consumers
# of already-degraded imagery must NOT harmonize again.)
NATIVE_QV = 65535.0   # S2-SHIPS (Raw) export scaling
CALIB_QV = 10000.0    # Appendix C calibration domain (reflectance x QV, QV=10000)


def to_calibration_domain(arr):
    """Harmonize a raw S2-SHIPS raster into the reflectance x 10000 domain."""
    return arr * (CALIB_QV / NATIVE_QV)


# Network input normalization: applied to the (already harmonized) values so
# they sit in roughly [0, 1]. In the calibration domain that divisor is CALIB_QV.
REFLECTANCE_SCALE = CALIB_QV

# Global base seed; per-sample seeds derive from hash(scene, band, severity, k).
BASE_SEED = 20260612


# ---------------------------------------------------------------------------
# Per-method hyperparameters (from Chapter 3 Section 3.5)
# ---------------------------------------------------------------------------
@dataclass
class TrainCfg:
    method: str
    patch: int
    batch_size: int
    epochs: int
    lr: float
    patches_per_epoch: int = 4000   # sampling budget; tune to scene count
    val_patches: int = 512
    num_workers: int = 6
    amp: bool = True                # mixed precision on the RTX 3090
    grad_clip: float = 0.0
    weight_decay: float = 0.0
    # self-supervised only:
    mask_rate: float = 0.198        # N2V / B2U blind-spot rate


DNCNN = TrainCfg(method="dncnn", patch=64, batch_size=128, epochs=100, lr=1e-3)
RESTORMER = TrainCfg(method="restormer", patch=128, batch_size=4, epochs=50, lr=1e-4,
                     patches_per_epoch=2000, amp=True)
NOISE2VOID = TrainCfg(method="noise2void", patch=64, batch_size=128, epochs=200, lr=4e-4)
BLIND2UNBLIND = TrainCfg(method="blind2unblind", patch=64, batch_size=128, epochs=200, lr=4e-4)

CFGS = {c.method: c for c in (DNCNN, RESTORMER, NOISE2VOID, BLIND2UNBLIND)}


def ckpt_dir(method: str, band: str, severity: str) -> Path:
    d = CHECKPOINT_ROOT / method / f"{band}_{severity}"
    d.mkdir(parents=True, exist_ok=True)
    return d
