"""
Data layer: resolve the train/eval scene partition, read single-band rasters,
and serve patch datasets for the supervised and self-supervised denoisers.

The denoisers train on the TRAINING partition only; the evaluation scenes are
held out (they are what the factorial degrades and runs YOLOv9m against).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C
from . import noise_model as nm

try:
    import rasterio  # noqa: F401
    _HAVE_RASTERIO = True
except Exception:  # pragma: no cover - only matters on Astraea
    _HAVE_RASTERIO = False


# ---------------------------------------------------------------------------
# Manifest / partition (CSV with per-band path columns)
# ---------------------------------------------------------------------------
def _load_manifest() -> dict:
    """Return {scene_id: row_dict} from the S2-SHIPS CSV manifest."""
    import csv
    if not C.MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"Manifest CSV not found at {C.MANIFEST_CSV}. Set MANIFEST_CSV in config.py."
        )
    with open(C.MANIFEST_CSV, newline="") as f:
        return {r["scene_id"]: r for r in csv.DictReader(f)}


def all_scene_ids() -> list[str]:
    return list(_load_manifest().keys())


def train_scene_ids() -> list[str]:
    """Denoiser training scenes = every scene NOT in the held-out EVAL set.

    The 4 EVAL scenes are the factorial scenes; holding them out of all
    denoiser training prevents leakage. The remaining 12 (detector train + val)
    are all usable for denoiser training since none are factorial scenes.
    """
    return [s for s in all_scene_ids() if s not in set(C.EVAL_SCENES)]


# ---------------------------------------------------------------------------
# Band raster reading (path comes straight from the manifest column)
# ---------------------------------------------------------------------------
def band_array_path(scene_id: str, band: str) -> str:
    row = _load_manifest()[scene_id]
    return row[C.BAND_PATH_COL[band]]


def read_band(scene_id: str, band: str) -> np.ndarray:
    """Read a single band, harmonized into the reflectance x 10000 calibration domain.

    The raw S2-SHIPS rasters are reflectance x 65535; harmonization (applied here,
    exactly once) brings them into the domain Appendix C calibrated against, so the
    Poisson-Gaussian noise is injected at the documented SNR.
    """
    if not _HAVE_RASTERIO:
        raise RuntimeError("rasterio is required on Astraea to read scene rasters.")
    import rasterio
    path = band_array_path(scene_id, band)
    if not Path(path).exists():
        raise FileNotFoundError(f"Band raster missing: {path}")
    with rasterio.open(path) as ds:
        raw = ds.read(1).astype(np.float32)  # single-band rasters per the manifest
    return C.to_calibration_domain(raw).astype(np.float32)


# In-memory cache so we don't re-read large rasters every patch.
_BAND_CACHE: dict[tuple[str, str], np.ndarray] = {}


def get_band(scene_id: str, band: str) -> np.ndarray:
    key = (scene_id, band)
    if key not in _BAND_CACHE:
        _BAND_CACHE[key] = read_band(scene_id, band)
    return _BAND_CACHE[key]


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------
def _sample_patch(arr: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = arr.shape
    if h < size or w < size:
        raise ValueError(f"Scene smaller ({h}x{w}) than patch {size}.")
    top = int(rng.integers(0, h - size + 1))
    left = int(rng.integers(0, w - size + 1))
    return arr[top:top + size, left:left + size]


def _augment(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        patch = np.fliplr(patch)
    if rng.random() < 0.5:
        patch = np.flipud(patch)
    k = int(rng.integers(0, 4))
    if k:
        patch = np.rot90(patch, k)
    return np.ascontiguousarray(patch)


# ---------------------------------------------------------------------------
# Supervised: clean/noisy pairs (DnCNN, Restormer) -- nominal severity
# ---------------------------------------------------------------------------
class SupervisedPairDataset(Dataset):
    """Yields (noisy, clean) normalized patches for a single band.

    Noisy targets are produced on the fly with the calibrated nominal-severity
    Poisson-Gaussian model. An epoch is a fixed number of randomly sampled
    patches across the training scenes.
    """

    def __init__(self, band: str, severity: str = "nominal", patch: int = 64,
                 length: int = 4000, augment: bool = True, seed_tag: str = "train"):
        self.band = band
        self.severity = severity
        self.patch = patch
        self.length = length
        self.augment = augment
        self.seed_tag = seed_tag
        self.scenes = train_scene_ids()
        if not self.scenes:
            raise RuntimeError("No training scenes resolved from manifest.")

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = nm.rng_for(self.seed_tag, self.band, self.severity, "pair", i)
        scene = self.scenes[int(rng.integers(0, len(self.scenes)))]
        clean = _sample_patch(get_band(scene, self.band), self.patch, rng)
        if self.augment:
            clean = _augment(clean, rng)
        noisy = nm.apply_poisson_gaussian(clean, self.band, self.severity, rng)
        clean = clean / C.REFLECTANCE_SCALE
        noisy = noisy / C.REFLECTANCE_SCALE
        return (torch.from_numpy(noisy)[None].float(),
                torch.from_numpy(clean)[None].float())


# ---------------------------------------------------------------------------
# Self-supervised: noisy-only (Noise2Void, Blind2Unblind) -- per severity
# ---------------------------------------------------------------------------
class NoisyOnlyDataset(Dataset):
    """Yields single noisy normalized patches for a band at a given severity.

    Noisy imagery is the training-partition clean scenes degraded at the target
    severity (deterministic). No clean reference is exposed.
    """

    def __init__(self, band: str, severity: str, patch: int = 64,
                 length: int = 4000, augment: bool = True, seed_tag: str = "train"):
        self.band = band
        self.severity = severity
        self.patch = patch
        self.length = length
        self.augment = augment
        self.seed_tag = seed_tag
        self.scenes = train_scene_ids()
        if not self.scenes:
            raise RuntimeError("No training scenes resolved from manifest.")

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = nm.rng_for(self.seed_tag, self.band, self.severity, "noisy", i)
        scene = self.scenes[int(rng.integers(0, len(self.scenes)))]
        clean = _sample_patch(get_band(scene, self.band), self.patch, rng)
        if self.augment:
            clean = _augment(clean, rng)
        noisy = nm.apply_poisson_gaussian(clean, self.band, self.severity, rng)
        noisy = noisy / C.REFLECTANCE_SCALE
        return torch.from_numpy(noisy)[None].float()
