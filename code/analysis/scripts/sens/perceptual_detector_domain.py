#!/usr/bin/env python3
# =============================================================================
# perceptual_detector_domain.py -- PSNR and SSIM in the detector's input domain
#
# Chapter 4 measures perceptual quality in the reflectance domain (reflectance
# in [0, 1], data_range = 1.0) against the clean reference, before the stretch
# and CLAHE that produce the 8-bit tiles the detector actually sees
# (evaluate_factorial.py). This script recomputes both metrics for every
# condition in the primary table (and the severity-matched table) in two
# domains:
#   refl : reflectance domain, exactly as the table (validation: must match)
#   det  : detector-input domain, after apply_stretch (stretch + CLAHE, uint8),
#          data_range = 255, the same transform applied to the clean reference
# and reports per-method deltas against the untreated baseline in both.
#
# It imports psnr/ssim/read_raster/paths from evaluate_factorial.py and
# apply_stretch/load_manifest/PATH_COL from build_detection_dataset.py, so the
# metric code and the stretch are the ones that produced Chapter 4. No GPU.
#
# USAGE (Astraea, from scripts/sens; the imports need ../ on sys.path, handled below):
#   python perceptual_detector_domain.py \
#       --table /data/praxis/results_p2/factorial_results.csv \
#       --matched-table /data/praxis/results_p2/matched/factorial_results.csv \
#       --manifest /data/praxis/processed/s2ships_manifest.csv \
#       --bounds /data/praxis/detection_tiled/stretch_bounds.json \
#       --degraded /data/praxis/synthetic-degradations/sentinel2 \
#       --denoised /data/praxis/denoised --denoised-matched /data/praxis/denoised_matched \
#       --out /data/praxis/results_sens/perceptual_domains
# =============================================================================
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from stats_sensitivity import bca_mean, wilcoxon_p, BASELINE_LABEL, CLEAN_LABEL

BLOCK = ["scene", "band", "severity", "realization"]

def load_backend():
    import evaluate_factorial as ef
    from build_detection_dataset import apply_stretch, load_manifest, PATH_COL
    return ef, apply_stretch, load_manifest, PATH_COL

def condition_native(ef, row, manifest, PATH_COL, degraded_root, denoised_root):
    """Native-domain (x 65535) array for one table row, as evaluate_factorial builds it."""
    m = manifest[row.scene]
    if row.method == CLEAN_LABEL:
        return ef.read_raster(m[PATH_COL[row.band]])
    if row.method == BASELINE_LABEL:
        src = ef.degraded_path(degraded_root, row.scene, row.band, row.severity, int(row.realization))
    else:
        src = ef.denoised_path(denoised_root, row.method, row.scene, row.band, row.severity, int(row.realization))
    if not os.path.exists(src): raise FileNotFoundError(src)
    return ef.read_raster(src) * ef.TO_NATIVE

def crop_like(a, b):
    if a.shape == b.shape: return a, b
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1]); return a[:h, :w], b[:h, :w]

def process(table, training, ef, apply_stretch, manifest, PATH_COL, bounds, degraded_root, denoised_root, cache):
    rows = []
    for i, r in enumerate(table.itertuples(index=False)):
        key = (r.scene, r.band)
        if key not in cache:
            clean_native = ef.read_raster(manifest[r.scene][PATH_COL[r.band]])
            lo, hi = bounds[r.band]["lo"], bounds[r.band]["hi"]
            cache[key] = dict(refl=clean_native / ef.NATIVE_QV, det=apply_stretch(clean_native, lo, hi).astype(np.float64), lo=lo, hi=hi)
        c = cache[key]
        try:
            arr = condition_native(ef, r, manifest, PATH_COL, degraded_root, denoised_root)
        except FileNotFoundError as e:
            print(f"[skip] {e}", flush=True); continue
        refl = arr / ef.NATIVE_QV; refl, cref = crop_like(refl, c["refl"])
        det = apply_stretch(arr, c["lo"], c["hi"]).astype(np.float64); det, cdet = crop_like(det, c["det"])
        rows.append(dict(training=training, method=r.method, band=r.band, severity=r.severity, realization=int(r.realization), scene=r.scene,
                         psnr_table=float(r.psnr) if str(r.psnr) != "inf" else np.inf, ssim_table=float(r.ssim),
                         psnr_refl=ef.psnr(refl, cref, 1.0), ssim_refl=ef.ssim(refl, cref, 1.0),
                         psnr_det=ef.psnr(det, cdet, 255.0), ssim_det=ef.ssim(det, cdet, 255.0),
                         mean_refl=float(refl.mean()), std_refl=float(refl.std()), std_det=float(det.std())))
        if (i + 1) % 50 == 0: print(f"  {i + 1}/{len(table)} ({training})", flush=True)
    return pd.DataFrame(rows)

def deltas(df, metrics):
    base = df[df.method == BASELINE_LABEL].set_index(BLOCK)
    out = []
    for m, g in df[~df.method.isin([BASELINE_LABEL, CLEAN_LABEL])].groupby("method"):
        g = g.set_index(BLOCK); b = base.reindex(g.index)
        for metric in metrics:
            d = (g[metric] - b[metric]).to_numpy(float); d = d[np.isfinite(d)]
            lo, hi = bca_mean(d)
            out.append(dict(method=m, metric=metric, n_blocks=int(d.size), mean_delta=float(d.mean()) if d.size else np.nan,
                            bca_lo=lo, bca_hi=hi, wilcoxon_p=wilcoxon_p(d)))
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True); ap.add_argument("--matched-table")
    ap.add_argument("--manifest", required=True); ap.add_argument("--bounds", required=True)
    ap.add_argument("--degraded", required=True); ap.add_argument("--denoised", required=True); ap.add_argument("--denoised-matched")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ef, apply_stretch, load_manifest, PATH_COL = load_backend()
    manifest = load_manifest(a.manifest); bounds = json.load(open(a.bounds))["bounds"]

    cache = {}
    t = pd.read_csv(a.table)
    res = process(t, "nominal", ef, apply_stretch, manifest, PATH_COL, bounds, a.degraded, a.denoised, cache)
    if a.matched_table and a.denoised_matched:
        mt = pd.read_csv(a.matched_table)
        res = pd.concat([res, process(mt, "matched", ef, apply_stretch, manifest, PATH_COL, bounds, a.degraded, a.denoised_matched, cache)], ignore_index=True)
    res.to_csv(out / "perceptual_both_domains.csv", index=False)

    # validation: the reflectance-domain recompute must reproduce the table
    fin = res[np.isfinite(res.psnr_table) & np.isfinite(res.psnr_refl)]
    dp = (fin.psnr_table - fin.psnr_refl).abs().max(); ds = (res.ssim_table - res.ssim_refl).abs().max()
    print(f"\nvalidation vs table: max |PSNR diff| = {dp:.4f} dB (table is rounded to 3 dp), max |SSIM diff| = {ds:.6f} (5 dp)")

    # levels: how saturated is SSIM in each domain?
    lvl = (res[res.training == "nominal"].groupby(["method"])[["psnr_refl", "psnr_det", "ssim_refl", "ssim_det", "std_refl", "std_det"]]
             .mean().round(4))
    print("\nmean level by method (nominal-trained rows):\n", lvl.to_string())

    # deltas vs the untreated baseline, pooled over 108 blocks, both domains
    nom = res[res.training == "nominal"]
    d_pooled = deltas(nom, ["psnr_refl", "psnr_det", "ssim_refl", "ssim_det"])
    d_pooled.to_csv(out / "perceptual_deltas_pooled.csv", index=False)
    print("\npooled deltas vs none (all blocks):\n", d_pooled.round(4).to_string(index=False))
    cells = []
    for (b, s), g in nom.groupby(["band", "severity"]):
        c = deltas(g, ["psnr_refl", "psnr_det", "ssim_refl", "ssim_det"]); c.insert(0, "severity", s); c.insert(0, "band", b); cells.append(c)
    pd.concat(cells, ignore_index=True).to_csv(out / "perceptual_deltas_per_cell.csv", index=False)

    # matched vs nominal-trained on both domains (H3 perceptual axis, both domains)
    if (res.training == "matched").any():
        keys = BLOCK + ["method"]
        j = (res[res.training == "matched"].set_index(keys)[["psnr_refl", "psnr_det", "ssim_refl", "ssim_det"]]
             - res[res.training == "nominal"].set_index(keys)[["psnr_refl", "psnr_det", "ssim_refl", "ssim_det"]]).dropna().reset_index()
        rows = []
        for m, g in j.groupby("method"):
            for metric in ["psnr_refl", "psnr_det", "ssim_refl", "ssim_det"]:
                d = g[metric].to_numpy(float); lo, hi = bca_mean(d)
                rows.append(dict(method=m, metric=metric, n_blocks=int(d.size), mean_delta=float(d.mean()), bca_lo=lo, bca_hi=hi, wilcoxon_p=wilcoxon_p(d)))
        mm = pd.DataFrame(rows); mm.to_csv(out / "perceptual_matched_vs_nominal.csv", index=False)
        print("\nmatched-trained minus nominal-trained:\n", mm.round(4).to_string(index=False))
    print("\nwritten to", out.resolve())

if __name__ == "__main__": main()
