#!/usr/bin/env python3
# =============================================================================
# matched_analysis.py -- severity-matched training arm (H3, Chapter 1 ¶506 form)
# per detector variant and seed.
#
# H3: a supervised denoiser trained at the severity at which it is evaluated
# achieves greater perceptual fidelity than the same architecture trained at
# nominal severity and evaluated under severity shift, without a corresponding
# gain in detection performance.
#
# Unit of analysis = matched block (scene x band x severity x realization),
# severities moderate and severe, methods dncnn and restormer. Three paired
# contrasts per block:
#   d_vs_nominal : matched-trained  minus nominal-trained   (the H3 contrast)
#   d_vs_none    : matched-trained  minus untreated baseline
#   shift_penalty: nominal-trained  minus untreated baseline (context)
# Detection axis per detector (mAP@0.5): pooled over 72 blocks per method and per
# band x severity cell (12 blocks); BCa 95% intervals (10,000 resamples, blocks
# resampled), Wilcoxon signed-rank, BH within detector, TOST at +/- RESOLUTION
# on the pooled H3 contrast. Perceptual axis (PSNR, SSIM) is detector-independent
# and is computed once from the seed-0 tables of the reference variant.
# Verdict per detector x method:
#   perceptual_gain        = BCa lower bound of dPSNR(matched - nominal) > 0
#   detection_gain_absent  = 95% BCa of d_vs_nominal includes 0
#   detection_equivalent   = 90% BCa of d_vs_nominal inside (-RES, +RES)
#   h3_supported           = perceptual_gain and detection_gain_absent
# Inference is read from the seed-0 detectors; seeds 1 and 2 are descriptive.
# Bootstrap, Wilcoxon, BH and RESOLUTION are imported from stats_sensitivity.py
# so the two arms share one implementation.
#
# USAGE (Astraea, from scripts/sens):
#   python assemble_sensitivity_table.py --root /data/praxis/results_sens \
#       --out /data/praxis/results_sens/all_variants_matched.csv \
#       --csv-name matched/factorial_results.csv --expect-conditions 12 --expect-treatments 2
#   python matched_analysis.py --table /data/praxis/results_sens/all_variants.csv \
#       --matched /data/praxis/results_sens/all_variants_matched.csv \
#       --out /data/praxis/results_sens/matched_out \
#       --primary-matched /data/praxis/results_p2/matched/factorial_results.csv
# =============================================================================
import argparse, json, sys, platform
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_sensitivity import bca_mean, bh, wilcoxon_p, RESOLUTION, N_BOOT, BOOT_SEED, BASELINE_LABEL, CLEAN_LABEL

MATCHED_METHODS = ["dncnn", "restormer"]
MATCHED_SEVS    = ["moderate", "severe"]
BLOCK = ["scene", "band", "severity", "realization"]
REFERENCE_VARIANT, REFERENCE_SEED = "m", 0

def norm_cols(d):
    ren = {}
    for c in d.columns:
        lc = c.lower()
        if lc == "map50": ren[c] = "mAP50"
        elif lc in ("map50_95", "map5095"): ren[c] = "mAP50_95"
        elif lc == "psnr": ren[c] = "PSNR"
        elif lc == "ssim": ren[c] = "SSIM"
    d = d.rename(columns=ren)
    for c in ("PSNR", "SSIM", "mAP50", "mAP50_95"):
        if c in d.columns: d[c] = pd.to_numeric(d[c].replace("inf", np.inf), errors="coerce")
    d["variant"] = d["variant"].astype(str); d["seed"] = d["seed"].astype(int)
    return d

def delta_rows(df, col, by):
    """Summarise paired deltas in `col` grouped by `by` (list of columns)."""
    out = []
    for keys, g in df.groupby(by, sort=False):
        x = g[col].to_numpy(float); x = x[np.isfinite(x)]
        lo, hi = bca_mean(x); row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row.update(n_blocks=int(x.size), mean=float(np.mean(x)) if x.size else np.nan,
                   median=float(np.median(x)) if x.size else np.nan, bca_lo=lo, bca_hi=hi, wilcoxon_p=wilcoxon_p(x))
        out.append(row)
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="assembled full-grid table (nominal-trained methods + none + clean)")
    ap.add_argument("--matched", required=True, help="assembled severity-matched table (dncnn, restormer at moderate/severe)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inference-seed", type=int, default=0)
    ap.add_argument("--primary-matched", help="Chapter 4 severity-matched table (m seed 0) for the reproduction check")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "run_log.txt", "w")
    def say(*s):
        t = " ".join(str(x) for x in s); print(t); log.write(t + "\n")

    full = norm_cols(pd.read_csv(a.table)); mt = norm_cols(pd.read_csv(a.matched))
    order = [v for v in ("m", "c", "e") if v in set(mt.variant)] + sorted(set(mt.variant) - {"m", "c", "e"})
    for d in (full, mt): d["variant"] = pd.Categorical(d["variant"], categories=order, ordered=True)
    say(f"full table rows: {len(full)}; matched rows: {len(mt)}; RESOLUTION = {RESOLUTION}")
    say("matched rows per variant x seed:\n", mt.groupby(["variant", "seed"], observed=True).size().unstack())
    bad_m = set(mt.method.unique()) - set(MATCHED_METHODS); bad_s = set(mt.severity.unique()) - set(MATCHED_SEVS)
    if bad_m or bad_s: sys.exit(f"matched table has unexpected labels: methods {bad_m}, severities {bad_s}")

    # --------------------------------------------------------------- pair the three frames per detector
    pairs = []
    for (v, s), m_ in mt.groupby(["variant", "seed"], observed=True):
        f = full[(full.variant == v) & (full.seed == s)]
        nom = f[f.method.isin(MATCHED_METHODS) & f.severity.isin(MATCHED_SEVS)]
        base = f[(f.method == BASELINE_LABEL) & f.severity.isin(MATCHED_SEVS)]
        keep = ["mAP50"] + [c for c in ("PSNR", "SSIM") if c in m_.columns and c in nom.columns and c in base.columns]
        j = (m_[BLOCK + ["method"] + keep].rename(columns={c: f"{c}_matched" for c in keep})
               .merge(nom[BLOCK + ["method"] + keep].rename(columns={c: f"{c}_nominal" for c in keep}), on=BLOCK + ["method"], how="inner")
               .merge(base[BLOCK + keep].rename(columns={c: f"{c}_none" for c in keep}), on=BLOCK, how="inner"))
        if len(j) != len(m_): say(f"[warn] {v} seed {s}: {len(m_)} matched rows but {len(j)} paired; unpaired rows dropped")
        j.insert(0, "seed", s); j.insert(0, "variant", v)
        for c in keep:
            j[f"d_{c}_vs_nominal"] = j[f"{c}_matched"] - j[f"{c}_nominal"]
            j[f"d_{c}_vs_none"]    = j[f"{c}_matched"] - j[f"{c}_none"]
            j[f"shift_penalty_{c}"] = j[f"{c}_nominal"] - j[f"{c}_none"]
        pairs.append(j)
    P = pd.concat(pairs, ignore_index=True); P["variant"] = pd.Categorical(P["variant"].astype(str), categories=order, ordered=True)
    P.to_csv(out / "matched_per_block.csv", index=False)

    # --------------------------------------------------------------- detection axis per detector
    det_pooled, det_cell = [], []
    for (v, s), g in P.groupby(["variant", "seed"], observed=True):
        tag = "inference" if s == a.inference_seed else "descriptive"
        say(f"\n==================== {v} seed {s} ({tag}) ====================")
        for contrast in ("d_mAP50_vs_nominal", "d_mAP50_vs_none", "shift_penalty_mAP50"):
            t = delta_rows(g, contrast, ["method"]); t.insert(0, "contrast", contrast); t.insert(0, "seed", s); t.insert(0, "variant", v)
            if contrast == "d_mAP50_vs_nominal":
                lo90, hi90, pt = [], [], []
                for m in t.method:
                    x = g[g.method == m][contrast].to_numpy(float); x = x[np.isfinite(x)]
                    l, h = bca_mean(x, conf=0.90); lo90.append(l); hi90.append(h)
                    p1 = stats.ttest_1samp(x, -RESOLUTION, alternative="greater").pvalue
                    p2 = stats.ttest_1samp(x,  RESOLUTION, alternative="less").pvalue
                    pt.append(float(max(p1, p2)))
                t["bca90_lo"] = lo90; t["bca90_hi"] = hi90; t["tost_p"] = pt
                t["tost_equivalent"] = np.isfinite(t.bca90_lo) & (t.bca90_lo > -RESOLUTION) & (t.bca90_hi < RESOLUTION)
            det_pooled.append(t)
            c = delta_rows(g, contrast, ["band", "severity", "method"]); c.insert(0, "contrast", contrast); c.insert(0, "seed", s); c.insert(0, "variant", v)
            det_cell.append(c)
        pooled_v = pd.concat([t for t in det_pooled if (t.variant == v).all() and (t.seed == s).all()], ignore_index=True)
        say(pooled_v.drop(columns=["variant", "seed"]).round(4).to_string(index=False))
    det_pooled = pd.concat(det_pooled, ignore_index=True); det_cell = pd.concat(det_cell, ignore_index=True)
    # BH within detector, across the H3 contrast tests (two methods) for the pooled table and across all
    # cells x methods for the per-cell table; the other two contrasts are context and are left uncorrected
    for df in (det_pooled, det_cell):
        df["p_bh"] = np.nan
        for (v, s), idx in df[df.contrast == "d_mAP50_vs_nominal"].groupby(["variant", "seed"], observed=True).groups.items():
            df.loc[idx, "p_bh"] = bh(df.loc[idx, "wilcoxon_p"].to_numpy())
    det_pooled.to_csv(out / "matched_detection_pooled.csv", index=False)
    det_cell.to_csv(out / "matched_detection_per_cell.csv", index=False)

    # --------------------------------------------------------------- perceptual axis (detector-independent): reference detector
    ref = P[(P.variant == REFERENCE_VARIANT) & (P.seed == REFERENCE_SEED)]
    if ref.empty: ref = P[P.seed == a.inference_seed].groupby("variant").get_group(sorted(P.variant.unique())[0])
    perc = []
    for metric in ("PSNR", "SSIM"):
        if f"d_{metric}_vs_nominal" not in ref.columns: say(f"[note] {metric} not present in both tables; perceptual axis skipped for it"); continue
        for contrast in (f"d_{metric}_vs_nominal", f"d_{metric}_vs_none", f"shift_penalty_{metric}"):
            t = delta_rows(ref, contrast, ["method"]); t.insert(0, "contrast", contrast); t.insert(0, "metric", metric); perc.append(t)
            c = delta_rows(ref, contrast, ["band", "severity", "method"]); c.insert(0, "contrast", contrast); c.insert(0, "metric", metric); c.insert(0, "scope", "cell"); perc.append(c)
    perc = pd.concat(perc, ignore_index=True) if perc else pd.DataFrame()
    if len(perc):
        perc["scope"] = perc.get("scope", pd.Series(index=perc.index, dtype=object)).fillna("pooled")
        perc.to_csv(out / "matched_perceptual.csv", index=False)
        say("\n==================== perceptual axis (detector-independent) ====================")
        say(perc[perc.scope == "pooled"].drop(columns=["scope", "band", "severity"], errors="ignore").round(4).to_string(index=False))

    # --------------------------------------------------------------- H3 verdicts per detector x method
    ver = []
    h3 = det_pooled[det_pooled.contrast == "d_mAP50_vs_nominal"]
    for _, r in h3.iterrows():
        pg = np.nan
        if len(perc):
            pp = perc[(perc.scope == "pooled") & (perc.metric == "PSNR") & (perc.contrast == "d_PSNR_vs_nominal") & (perc.method == r.method)]
            if len(pp): pg = bool(np.isfinite(pp.bca_lo.iloc[0]) and pp.bca_lo.iloc[0] > 0)
        includes0 = bool(np.isfinite(r.bca_lo) and r.bca_lo <= 0 <= r.bca_hi)
        # a detection gain counts by the same rule as everywhere else in the praxis: above resolution and BCa lower bound > 0
        gain = bool(r["mean"] > RESOLUTION and np.isfinite(r.bca_lo) and r.bca_lo > 0)
        ver.append(dict(variant=r.variant, seed=r.seed, method=r.method, role="inference" if r.seed == a.inference_seed else "descriptive",
                        dmAP_matched_vs_nominal=r["mean"], bca_lo=r.bca_lo, bca_hi=r.bca_hi, p_bh=r.p_bh,
                        perceptual_gain=pg, detection_ci_includes_zero=includes0, detection_beats_resolution=gain,
                        detection_equivalent=bool(r.tost_equivalent),
                        h3_supported=(bool(pg) and not gain) if pg is not np.nan else np.nan))
    ver = pd.DataFrame(ver); ver.to_csv(out / "matched_h3_verdicts.csv", index=False)
    say("\n==================== H3 verdicts ====================")
    say(ver.round(4).to_string(index=False))

    # --------------------------------------------------------------- reproduction check against the Chapter 4 matched table
    if a.primary_matched:
        prim = norm_cols(pd.read_csv(a.primary_matched).assign(variant=REFERENCE_VARIANT, seed=REFERENCE_SEED))
        m0 = mt[(mt.variant == REFERENCE_VARIANT) & (mt.seed == a.inference_seed)]
        j = m0.merge(prim[BLOCK + ["method", "mAP50"]].rename(columns={"mAP50": "primary"}), on=BLOCK + ["method"], how="inner")
        j["abs_diff"] = (j.mAP50 - j.primary).abs()
        say(f"\nReproduction check (matched, {REFERENCE_VARIANT} seed {a.inference_seed}): matched rows {len(j)} of {len(m0)}; "
            f"max |diff| = {j.abs_diff.max():.6f}; rows with |diff| > 1e-6: {int((j.abs_diff > 1e-6).sum())}")
        j.to_csv(out / "reproduction_check_matched_m_seed0.csv", index=False)

    (out / "sessionInfo.json").write_text(json.dumps(dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        scipy=__import__("scipy").__version__, host=platform.node(), n_boot=N_BOOT, boot_seed=BOOT_SEED, resolution=RESOLUTION,
        inference_seed=a.inference_seed), indent=2))
    say("\nOutputs written to", out.resolve()); log.close()

if __name__ == "__main__": main()
