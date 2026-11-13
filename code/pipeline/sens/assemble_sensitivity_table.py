#!/usr/bin/env python3
# =============================================================================
# assemble_sensitivity_table.py -- concatenate the nine sweep tables into one
# all_variants.csv with `variant` and `seed` columns, validate the result, and
# print the exact labels to paste into the CONFIG block of both analysis scripts.
#
# USAGE (Astraea, env pacific):
#   python assemble_sensitivity_table.py \
#       --root /data/praxis/results_sens \
#       --out  /data/praxis/results_sens/all_variants.csv \
#       --expect-conditions 36          # or 54 if the arms ran the full grid
#
# Expects one directory per run under --root, named as in the handoff:
#   tiled_p2 tiled_p2_s1 tiled_p2_s2 tiled_p2_c tiled_p2_c_s1 tiled_p2_c_s2
#   tiled_p2_e tiled_p2_e_s1 tiled_p2_e_s2
# each containing factorial_results.csv (override with --csv-name).
# If the sweep already emits `variant` and `seed` columns they are checked, not
# overwritten. Column names are normalised to the schema the analysis scripts
# read: scene, band, severity, realization, method, mAP50, mAP50_95.
# =============================================================================
import argparse, sys, re
from pathlib import Path
import pandas as pd

RUNS = {  # run directory -> (variant, seed)
    "tiled_p2": ("m", 0), "tiled_p2_s1": ("m", 1), "tiled_p2_s2": ("m", 2),
    "tiled_p2_c": ("c", 0), "tiled_p2_c_s1": ("c", 1), "tiled_p2_c_s2": ("c", 2),
    "tiled_p2_e": ("e", 0), "tiled_p2_e_s1": ("e", 1), "tiled_p2_e_s2": ("e", 2),
}
# accepted aliases for each schema column (lower-cased, non-alphanumerics stripped)
ALIASES = {
    "scene": ["scene", "scene_id", "scenename", "site"],
    "band": ["band", "band_id", "channel"],
    "severity": ["severity", "severity_level", "level", "noise_level"],
    "realization": ["realization", "realisation", "noise_seed", "noise_realization", "rep", "replicate"],
    "method": ["method", "treatment", "denoiser", "condition_method"],
    "mAP50": ["map50", "map05", "mapat05", "map_50", "metricsmap50b", "map@0.5", "map0.5"],
    "mAP50_95": ["map5095", "map50_95", "map0595", "map_50_95", "metricsmap5095b", "map@0.5:0.95", "map0.50.95"],
    "PSNR": ["psnr"], "SSIM": ["ssim"],
}
def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())

def map_columns(df):
    present = {norm(c): c for c in df.columns}
    out = {}
    for target, names in ALIASES.items():
        for n in names:
            if norm(n) in present: out[target] = present[norm(n)]; break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--csv-name", default="factorial_results.csv")
    ap.add_argument("--expect-conditions", type=int, default=36,
                    help="band x severity x treatment cells: 36 for the 3.8.7 subset (3 x 2 x 6), 54 for the full grid (3 x 3 x 6)")
    ap.add_argument("--expect-scenes", type=int, default=4); ap.add_argument("--expect-realizations", type=int, default=3)
    ap.add_argument("--expect-treatments", type=int, default=6, help="including the untreated baseline")
    a = ap.parse_args()
    root = Path(a.root); frames = []; problems = []
    for run, (variant, seed) in RUNS.items():
        f = root / run / a.csv_name
        if not f.exists(): problems.append(f"MISSING  {f}"); continue
        df = pd.read_csv(f); cm = map_columns(df)
        need = ["scene", "band", "severity", "realization", "method", "mAP50"]
        miss = [c for c in need if c not in cm]
        if miss: problems.append(f"{run}: cannot find columns {miss}; have {list(df.columns)}"); continue
        df = df.rename(columns={v: k for k, v in cm.items()})
        keep = [c for c in ["scene", "band", "severity", "realization", "method", "mAP50", "mAP50_95", "PSNR", "SSIM"] if c in df.columns]
        df = df[keep].copy()
        if "variant" in df.columns and (df["variant"].astype(str) != variant).any():
            problems.append(f"{run}: file says variant {df['variant'].unique()} but directory implies {variant}")
        if "seed" in df.columns and (df["seed"].astype(int) != seed).any():
            problems.append(f"{run}: file says seed {df['seed'].unique()} but directory implies {seed}")
        df.insert(0, "seed", seed); df.insert(0, "variant", variant); df.insert(0, "run", run)
        frames.append(df)
    if problems:
        print("PROBLEMS:"); [print("  ", p) for p in problems]
        if not frames: sys.exit(1)
    all_ = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------- validation
    print("\n=== rows per run ==="); print(all_.groupby("run").size().to_string())
    print("\n=== distinct labels ===")
    for c in ["scene", "band", "severity", "method"]:
        print(f"  {c:12s}: {sorted(all_[c].astype(str).unique())}")
    print(f"  realization : {sorted(all_['realization'].unique())}")

    methods = sorted(all_["method"].astype(str).unique())
    guess_base = [m for m in methods if norm(m) in ("none", "baseline", "degraded", "untreated", "nodenoise", "nodenoising", "raw")]
    guess_clean = [m for m in methods if norm(m) in ("clean", "reference", "cleanreference", "ceiling", "clean_ref")]
    print("\n=== CONFIG labels (paste into both analysis scripts) ===")
    print(f'  BASELINE_LABEL = "{guess_base[0] if guess_base else "<SET: one of " + str(methods) + ">"}"')
    print(f'  CLEAN_LABEL    = "{guess_clean[0] if guess_clean else "<SET or confirm clean rows are absent>"}"')
    sevs = sorted(all_["severity"].astype(str).unique()); bands = sorted(all_["band"].astype(str).unique())
    print(f'  SEVERE_LABEL   = "{[s for s in sevs if "sev" in s.lower()][0] if any("sev" in s.lower() for s in sevs) else "<SET: one of " + str(sevs) + ">"}"')
    print(f'  B05_LABEL      = "{[b for b in bands if "05" in b or b.lower()=="b5"][0] if any("05" in b or b.lower()=="b5" for b in bands) else "<SET: one of " + str(bands) + ">"}"')

    # expected counts
    fact = all_[~all_["method"].astype(str).str.lower().isin([norm(m) for m in guess_clean] + ["clean"])]
    exp_rows = a.expect_conditions * a.expect_scenes * a.expect_realizations
    print(f"\n=== expected treatment rows per run: {a.expect_conditions} conditions (band x severity x treatment) x "
          f"{a.expect_scenes} scenes x {a.expect_realizations} realizations = {exp_rows} ===")
    bad = []
    for run, g in fact.groupby("run"):
        n = len(g); flag = "" if n == exp_rows else f"   <-- expected {exp_rows}"
        print(f"  {run:14s} {n:6d}{flag}")
        if n != exp_rows: bad.append(run)
        dup = g.duplicated(["scene", "band", "severity", "realization", "method"]).sum()
        if dup: print(f"     DUPLICATE block/method rows: {dup}"); bad.append(run)
    nclean = len(all_) - len(fact)
    print(f"\n  clean-reference rows total: {nclean}" + ("" if nclean else "   <-- NONE: headroom cannot be computed; re-run sweeps with the clean condition"))
    # per-block completeness: every block should have all treatments
    cnt = fact.groupby(["run", "scene", "band", "severity", "realization"]).size()
    incomplete = cnt[cnt != a.expect_treatments]
    if len(incomplete): print(f"\n  INCOMPLETE blocks (treatments != {a.expect_treatments}): {len(incomplete)}"); print(incomplete.head(20).to_string())
    all_.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(all_)} rows, {all_['run'].nunique()} runs")
    if bad or problems or len(incomplete): print("\nVALIDATION: issues found (see above). Do not run the analysis until they are resolved."); sys.exit(2)
    print("\nVALIDATION: clean.")

if __name__ == "__main__": main()
