"""
Run this FIRST on Astraea, before any training:

    python -m denoising.audit_data

It checks every assumption the training scripts rely on and prints what it
actually finds, so any path/layout mismatch surfaces here (cheaply) instead of
mid-training. Nothing is modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import config as C
from . import noise_model as nm


def _ok(msg): print(f"  [ok]   {msg}")
def _warn(msg): print(f"  [WARN] {msg}")
def _fail(msg): print(f"  [FAIL] {msg}")


def main():
    problems = 0
    print("=== Praxis denoiser data audit ===\n")

    # 1) Calibration
    print("Noise calibration:")
    if C.NOISE_CALIBRATION_JSON.exists():
        _ok(f"found {C.NOISE_CALIBRATION_JSON}")
    else:
        _warn(f"{C.NOISE_CALIBRATION_JSON} missing -> using Appendix C fallbacks")
    for b in C.BANDS:
        for sev in C.SEVERITIES:
            a, s = nm.params(b, sev)
            print(f"        {b}/{sev:8s}  alpha={a:9.4f}  sigma_r={s:6.3f}")

    # 2) Manifest / partition
    print("\nManifest / partition:")
    try:
        from .data import train_scene_ids
        scenes = train_scene_ids()
        if scenes:
            _ok(f"{len(scenes)} training scenes resolved (e.g. {scenes[:3]})")
        else:
            _fail("manifest resolved zero training scenes")
            problems += 1
    except Exception as e:
        _fail(f"could not resolve training scenes: {e}")
        problems += 1
        scenes = []

    # 3) Rasterio + a sample read per band
    print("\nScene rasters:")
    try:
        import rasterio  # noqa: F401
        _ok("rasterio importable")
    except Exception:
        _fail("rasterio not importable in this env (needed on Astraea)")
        problems += 1
        print("\nAudit stopped early (no rasterio).")
        sys.exit(1 if problems else 0)

    from .data import band_array_path, read_band, all_scene_ids
    if scenes:
        print(f"        partition: {len(all_scene_ids())} total, "
              f"{len(scenes)} train (held out EVAL: {', '.join(C.EVAL_SCENES)})")
        s0 = scenes[0]
        for b in C.BANDS:
            path = band_array_path(s0, b)
            try:
                arr = read_band(s0, b)
                lo, hi, mean = float(arr.min()), float(arr.max()), float(arr.mean())
                _ok(f"{b}: {Path(path).name} shape={arr.shape} "
                    f"range=({lo:.1f},{hi:.1f}) mean={mean:.1f}")
                if hi <= 1.5:
                    _warn(f"{b} values look like reflectance (<=1.5); "
                          f"REFLECTANCE_SCALE={C.REFLECTANCE_SCALE} may need to be 1.0")
                elif hi > 20000:
                    _warn(f"{b} max {hi:.0f} unusually high; confirm DN scaling")
            except Exception as e:
                _fail(f"{b}: cannot read {path}: {e}")
                problems += 1

    # 4) Patch feasibility
    print("\nPatch sizes:")
    if scenes:
        try:
            arr = read_band(scenes[0], C.BANDS[0])
            for cfg in (C.DNCNN, C.RESTORMER, C.NOISE2VOID, C.BLIND2UNBLIND):
                fits = min(arr.shape) >= cfg.patch
                (_ok if fits else _fail)(f"{cfg.method}: patch {cfg.patch} "
                                         f"{'fits' if fits else 'TOO LARGE for'} "
                                         f"smallest dim {min(arr.shape)}")
                problems += 0 if fits else 1
        except Exception as e:
            _warn(f"could not check patch feasibility: {e}")

    # 5) Smoke test the noise model output stats
    print("\nNoise model sanity (synthetic flat field, DN=1000):")
    flat = np.full((128, 128), 1000.0, np.float32)
    for b in C.BANDS:
        y = nm.degrade(flat, "AUDIT", b, "nominal", 0)
        a, s = nm.params(b, "nominal")
        expected_var = 1000.0 / a + s * s
        print(f"        {b}: emp_var={y.var():.1f}  expected~{expected_var:.1f}")

    # 6) Harmonization check: SNR at a real harmonized pixel should match the spec
    print("\nHarmonization / SNR check (real imagery, harmonized to calib domain):")
    if scenes:
        try:
            for b in C.BANDS:
                arr = read_band(scenes[0], b)  # already harmonized
                I = float(np.median(arr))
                a, s = nm.params(b, "nominal")
                snr = I / np.sqrt(I / a + s * s)
                print(f"        {b}: harmonized median={I:8.1f}  SNR_at_median={snr:6.1f}")
            print("        (compare to ESA spec SNR ~142/117/100; harmonized medians "
                  "should now sit near the calibration P_ref scale)")
        except Exception as e:
            _warn(f"could not run SNR check: {e}")

    print("\n=== Audit complete:",
          "ALL GOOD" if problems == 0 else f"{problems} problem(s) to fix",
          "===")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
