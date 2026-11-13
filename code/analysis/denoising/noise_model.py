"""
Poisson-Gaussian noise model (Appendix C).

    y = (1/alpha) * Poisson(alpha * I) + N(0, sigma_r^2)

with I, y in DN units. Per-band (alpha, sigma_r) come from
noise_calibration.json when present, else from config fallbacks scaled by the
Appendix C severity factors. Seeding follows the project scheme
hash(scene_id, band, severity, realization) and uses hashlib (NOT Python's
salted hash()) so seeds are stable across processes and runs.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np

from . import config as C


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------
def seed_from(*parts) -> int:
    """Stable 32-bit seed from arbitrary parts, e.g. (scene, band, sev, k)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1)


def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(seed_from(C.BASE_SEED, *parts))


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_calibration() -> dict:
    """Return {band: {severity: {'alpha':..., 'sigma_r':...}}}.

    Tolerant of two JSON shapes:
      (a) per-(band,severity) already enumerated, e.g.
          {"B04": {"nominal": {"alpha":.., "sigma_r":..}, "moderate": {...}, ...}, ...}
      (b) per-band nominal + global severity_factors, e.g.
          {"bands": {"B04": {"alpha":.., "sigma_r":..}, ...},
           "severity_factors": {"moderate": {"alpha_factor":.., "sigma_factor":..}, ...}}
    Falls back to config.NOMINAL_FALLBACK + config.SEVERITY_FACTORS if the file
    is missing.
    """
    path = C.NOISE_CALIBRATION_JSON
    if path.exists():
        raw = json.loads(path.read_text())
    else:
        raw = None

    out: dict[str, dict[str, dict[str, float]]] = {b: {} for b in C.BANDS}

    def expand(nominal: dict[str, dict[str, float]], factors: dict[str, dict[str, float]]):
        for b in C.BANDS:
            base = nominal[b]
            for sev in C.SEVERITIES:
                f = factors[sev]
                out[b][sev] = {
                    "alpha": base["alpha"] * f["alpha_factor"],
                    "sigma_r": base["sigma_r"] * f["sigma_factor"],
                }

    if raw is None:
        expand(C.NOMINAL_FALLBACK, C.SEVERITY_FACTORS)
        return out

    # Shape (a): top-level keys are band names with per-severity dicts.
    if all(b in raw for b in C.BANDS) and isinstance(raw[C.BANDS[0]], dict) \
            and any(s in raw[C.BANDS[0]] for s in C.SEVERITIES):
        for b in C.BANDS:
            for sev in C.SEVERITIES:
                out[b][sev] = {
                    "alpha": float(raw[b][sev]["alpha"]),
                    "sigma_r": float(raw[b][sev]["sigma_r"]),
                }
        return out

    # Shape (b): bands + (optional) severity_factors. Tolerant of the
    # Appendix C key names ('alpha_calibrated', 'read_noise_sigma_r') as well as
    # the short forms ('alpha', 'sigma_r').
    src = raw.get("bands", raw)

    def _alpha(d):
        for k in ("alpha", "alpha_calibrated"):
            if k in d:
                return float(d[k])
        raise KeyError("no alpha/alpha_calibrated in band entry")

    def _sigma(d):
        for k in ("sigma_r", "read_noise_sigma_r"):
            if k in d:
                return float(d[k])
        raise KeyError("no sigma_r/read_noise_sigma_r in band entry")

    nominal = {b: {"alpha": _alpha(src[b]), "sigma_r": _sigma(src[b])} for b in C.BANDS}

    # Severity scaling: prefer the JSON's "severity_levels" block, because that
    # is exactly what apply_degradation.py uses (calib["severity_levels"][sev]
    # ["alpha_scale"|"sigma_r_scale"]). Reading the same source guarantees the
    # denoisers train on the identical per-severity noise the factorial injects.
    if "severity_levels" in raw:
        lv = raw["severity_levels"]
        for b in C.BANDS:
            for sev in C.SEVERITIES:
                out[b][sev] = {
                    "alpha": nominal[b]["alpha"] * float(lv[sev]["alpha_scale"]),
                    "sigma_r": nominal[b]["sigma_r"] * float(lv[sev]["sigma_r_scale"]),
                }
        return out

    # Otherwise fall back to a "severity_factors" block or the config defaults.
    factors = raw.get("severity_factors", C.SEVERITY_FACTORS)
    factors = {sev: {"alpha_factor": float(factors[sev]["alpha_factor"]),
                     "sigma_factor": float(factors[sev]["sigma_factor"])}
               for sev in C.SEVERITIES}
    expand(nominal, factors)
    return out


def params(band: str, severity: str) -> tuple[float, float]:
    """Return (alpha, sigma_r) in DN for a band/severity."""
    p = _load_calibration()[band][severity]
    return p["alpha"], p["sigma_r"]


# ---------------------------------------------------------------------------
# Noise application
# ---------------------------------------------------------------------------
def apply_poisson_gaussian(clean_dn: np.ndarray, band: str, severity: str,
                           rng: np.random.Generator) -> np.ndarray:
    """Apply the calibrated Poisson-Gaussian model to a DN array.

    clean_dn : float array in DN units (>= 0 expected; negatives are clipped to 0
               before the Poisson draw).
    Returns a float array in DN units (may contain small negatives from read
    noise; left unclipped so the statistics match Appendix C).
    """
    alpha, sigma_r = params(band, severity)
    lam = np.clip(clean_dn, 0.0, None) * alpha
    shot = rng.poisson(lam).astype(np.float64) / alpha
    read = rng.normal(0.0, sigma_r, size=clean_dn.shape)
    return (shot + read).astype(np.float32)


def degrade(clean_dn: np.ndarray, scene_id: str, band: str, severity: str,
            realization: int = 0) -> np.ndarray:
    """Deterministic degradation seeded by the project scheme."""
    rng = rng_for(scene_id, band, severity, realization)
    return apply_poisson_gaussian(clean_dn, band, severity, rng)
