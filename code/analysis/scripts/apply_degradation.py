#!/usr/bin/env python3
"""
apply_degradation.py — Synthetic degradation for the S2-SHIPS factorial (Chapter 3, Sec. 3.4).

Applies the calibrated Poisson-Gaussian noise model

    y = (1/alpha_eff) * Poisson(alpha_eff * I) + N(0, sigma_eff^2)

in pixel-value units (reflectance x QV) to each clean Sentinel-2 band image, across the
full factorial of scenes x bands x severities x realizations.

Per band, per severity:
    alpha_eff = alpha_calibrated[band] * alpha_scale[severity]
    sigma_eff = read_noise_sigma_r[band] * sigma_r_scale[severity]

Determinism: each (scene, band, severity, realization) draws from a NumPy Generator seeded
by a stable SHA-256 of those four fields, so the entire set regenerates bit-for-bit
regardless of process or machine (independent of PYTHONHASHSEED).

Inputs : s2ships_manifest.csv, noise_calibration.json
Output : one GeoTIFF per (scene, band, severity, realization) + degradation_manifest.csv
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import rasterio

BANDS = ["B04", "B05", "B11"]
SEVERITIES = ["nominal", "moderate", "severe"]
PATH_COL = {"B04": "path_B04", "B05": "path_B05", "B11": "path_B11"}


def stable_seed(scene_id, band, severity, realization):
    """Process-independent 64-bit seed from the four factorial coordinates."""
    key = f"{scene_id}|{band}|{severity}|{realization}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def degrade(clean, alpha_eff, sigma_eff, rng):
    """Apply the Poisson-Gaussian model. `clean` is float; returns float, unclipped."""
    lam = alpha_eff * np.clip(clean, 0.0, None)          # Poisson rate must be >= 0
    shot = rng.poisson(lam).astype(np.float64) / alpha_eff
    read = rng.normal(0.0, sigma_eff, size=clean.shape)
    return shot + read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--realizations", type=int, default=3)
    args = ap.parse_args()

    with open(args.calib) as f:
        calib = json.load(f)
    sev_levels = calib["severity_levels"]
    bands_cal = calib["bands"]

    with open(args.manifest, newline="") as f:
        scenes = list(csv.DictReader(f))

    n_expected = len(scenes) * len(BANDS) * len(SEVERITIES) * args.realizations
    print(f"[plan] {len(scenes)} scenes x {len(BANDS)} bands x {len(SEVERITIES)} severities "
          f"x {args.realizations} realizations = {n_expected} images", flush=True)

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "degradation_manifest.csv")
    index_fields = ["scene_id", "band", "severity", "realization", "alpha_eff",
                    "sigma_eff", "seed", "source_path", "output_path"]
    written = 0
    with open(index_path, "w", newline="") as idx:
        w = csv.DictWriter(idx, fieldnames=index_fields)
        w.writeheader()
        for scene in scenes:
            sid = scene["scene_id"]
            for band in BANDS:
                src = scene[PATH_COL[band]]
                with rasterio.open(src) as ds:
                    clean = ds.read(1).astype(np.float64)
                    clean = clean * (10000.0 / 65535.0)  # harmonize: refl x65535 -> calib domain x10000
                    profile = ds.profile.copy()
                dtype = profile["dtype"]
                dmax = float(np.iinfo(dtype).max) if np.issubdtype(np.dtype(dtype), np.integer) else None
                a_cal = bands_cal[band]["alpha_calibrated"]
                s_cal = bands_cal[band]["read_noise_sigma_r"]
                for sev in SEVERITIES:
                    a_eff = a_cal * sev_levels[sev]["alpha_scale"]
                    s_eff = s_cal * sev_levels[sev]["sigma_r_scale"]
                    for r in range(args.realizations):
                        seed = stable_seed(sid, band, sev, r)
                        rng = np.random.default_rng(seed)
                        y = degrade(clean, a_eff, s_eff, rng)
                        if dmax is not None:
                            y = np.clip(np.rint(y), 0.0, dmax)
                        y = y.astype(dtype)
                        out_dir = os.path.join(args.out, sid, band, sev)
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"r{r}.tif")
                        with rasterio.open(out_path, "w", **profile) as dst:
                            dst.write(y, 1)
                        w.writerow({"scene_id": sid, "band": band, "severity": sev,
                                    "realization": r, "alpha_eff": f"{a_eff:.6f}",
                                    "sigma_eff": f"{s_eff:.6f}", "seed": seed,
                                    "source_path": src, "output_path": out_path})
                        written += 1
            print(f"[done] scene {sid} ({written} images so far)", flush=True)

    print(f"[total] wrote {written} images (expected {n_expected})", flush=True)
    print(f"[index] {index_path}", flush=True)
    if written != n_expected:
        sys.exit(f"ERROR: produced {written} images, expected {n_expected}")


if __name__ == "__main__":
    main()
