#!/usr/bin/env python3
"""
bm3d_denoise.py — Classical BM3D denoising for the evaluation factorial (Chapter 3, §3.5.1).

For each degraded image of the four EVALUATION scenes, apply the Poisson-Gaussian
denoising pipeline:

    forward GAT  ->  BM3D (sigma_psd = 1)  ->  closed-form unbiased inverse GAT

The generalized Anscombe transform (GAT) stabilizes the Poisson-Gaussian variance to ~1,
BM3D denoises the stabilized data, and the exact unbiased inverse (Makitalo & Foi 2013,
closed-form approximation) maps back to pixel-value units. The model y = (1/alpha)*
Poisson(alpha*I) + N(0, sigma_r^2) is standardized to unit gain by z = alpha*y, so the
transform operates on z ~ Poisson(alpha*I) + N(0, (alpha*sigma_r)^2) and the denoised
result is divided by alpha to return to pixel-value units. alpha_eff and sigma_eff are
read per image from the degradation manifest.

Scope: 4 eval scenes x 3 bands x 3 severities x 3 realizations = 108 images.

GAT formulas follow the Foi/Makitalo reference implementation accompanying
"Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian
noise," IEEE TIP 22(1):91-103, 2013.
"""
import argparse, csv, os, sys
import numpy as np
import rasterio
from bm3d import bm3d

EVAL_SCENES = {"panama", "rome", "brest1", "rotterdam1"}
_SQRT_3_2 = np.sqrt(3.0 / 2.0)

# The closed-form unbiased inverse GAT (Makitalo & Foi 2013) is derived for the
# STANDARDIZED model z ~ Poisson(lambda) + N(0, sigma_z^2) (unit gain). Our model
# y = (1/alpha)*Poisson(alpha*I) + N(0, sigma_r^2) is standardized by z = alpha*y,
# giving lambda = alpha*I and sigma_z = alpha*sigma_r. We transform in z-space and
# divide the result by alpha to return to pixel-value units.


def gat_forward_std(z, sigma_z):
    """Standardized (unit-gain) generalized Anscombe forward; output variance ~1."""
    return 2.0 * np.sqrt(np.maximum(z + 3.0 / 8.0 + sigma_z ** 2, 0.0))


def gat_inverse_std(x, sigma_z):
    """Closed-form approximation of the exact unbiased inverse GAT (unit gain)."""
    t = np.maximum(x, 1.0)
    inv = (np.power(t / 2.0, 2.0)
           + 0.25 * _SQRT_3_2 * np.power(t, -1.0)
           - (11.0 / 8.0) * np.power(t, -2.0)
           + (5.0 / 8.0) * _SQRT_3_2 * np.power(t, -3.0)
           - 1.0 / 8.0 - sigma_z ** 2)
    inv = np.maximum(0.0, inv)
    inv[inv != inv] = 0.0          # NaN guard
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degraded-manifest", required=True,
                    help="degradation_manifest.csv produced by apply_degradation.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.degraded_manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["scene_id"] in EVAL_SCENES]

    n_expected = len(EVAL_SCENES) * 3 * 3 * 3
    print(f"[plan] {len(rows)} eval-partition images to denoise (expected {n_expected})", flush=True)
    if len(rows) != n_expected:
        sys.exit(f"ERROR: manifest yielded {len(rows)} eval rows, expected {n_expected}")

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "bm3d_manifest.csv")
    fields = ["scene_id", "band", "severity", "realization", "alpha_eff", "sigma",
              "source_path", "output_path"]
    written = 0
    with open(index_path, "w", newline="") as idx:
        w = csv.DictWriter(idx, fieldnames=fields); w.writeheader()
        for r in rows:
            sid, band, sev, real = r["scene_id"], r["band"], r["severity"], r["realization"]
            alpha_eff, sigma_eff = float(r["alpha_eff"]), float(r["sigma_eff"])
            src = r["output_path"]                       # the degraded image is BM3D's input
            with rasterio.open(src) as ds:
                noisy = ds.read(1).astype(np.float64)
                profile = ds.profile.copy()

            # standardize to unit-gain Poisson-Gaussian, transform, denoise, invert, rescale
            z = alpha_eff * noisy
            sigma_z = alpha_eff * sigma_eff
            stab = gat_forward_std(z, sigma_z)
            den_stab = bm3d(stab, sigma_psd=1.0)         # stabilized -> unit-variance noise
            den = gat_inverse_std(den_stab, sigma_z) / alpha_eff

            dtype = profile["dtype"]
            if np.issubdtype(np.dtype(dtype), np.integer):
                den = np.clip(np.rint(den), 0, np.iinfo(dtype).max)
            den = den.astype(dtype)

            out_dir = os.path.join(args.out, sid, band, sev)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"r{real}.tif")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(den, 1)
            w.writerow({"scene_id": sid, "band": band, "severity": sev, "realization": real,
                        "alpha_eff": f"{alpha_eff:.6f}", "sigma": f"{sigma_eff:.6f}",
                        "source_path": src, "output_path": out_path})
            written += 1
            print(f"[done] {sid}/{band}/{sev}/r{real}  ({written}/{n_expected})", flush=True)

    print(f"[total] wrote {written} denoised images (expected {n_expected})")
    print(f"[index] {index_path}")
    if written != n_expected:
        sys.exit(f"ERROR: produced {written}, expected {n_expected}")


if __name__ == "__main__":
    main()
