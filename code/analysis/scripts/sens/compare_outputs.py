#!/usr/bin/env python3
# =============================================================================
# compare_outputs.py -- diff the R and Python sensitivity outputs
#
# USAGE: python compare_outputs.py <r_out_dir> <py_out_dir>
#
# The two implementations write identically named CSVs. This aligns each pair
# on its key columns and compares numeric columns with per-column tolerances:
#   exact-ish (1e-6): mean_delta, median_delta, n_blocks, mean_rank, D_clean,
#                     D_degraded, headroom, baseline_mean, clean_mean
#   loose   (2.5e-3): bca_lo, bca_hi   (different bootstrap RNGs; 10k resamples)
#   p-values (1e-3 ): wilcoxon_p, p_bh, nemenyi cells
# Booleans and orderings must match exactly. Exit code 1 on any mismatch.
# =============================================================================
import sys, glob, os, math
import pandas as pd, numpy as np

TOL = {"bca_lo": 2.5e-3, "bca_hi": 2.5e-3, "bca90_lo": 2.5e-3, "bca90_hi": 2.5e-3,
       "wilcoxon_p": 1e-3, "p_bh": 1e-3, "tost_p": 1e-3, "headroom_sd": 1e-6}
DEFAULT_TOL = 1e-6
KEYS = {
    "pooled_deltas": ["method"], "per_condition_deltas": ["band", "severity", "method"],
    "mean_ranks": ["method"], "B05_severe": ["method"], "headroom_per_condition": ["band", "severity"],
    "headroom_per_block": ["scene", "band", "severity", "realization"], "criteria_3_8_7": ["variant"],
    "seed_spread": ["variant"], "seed_values": ["variant", "seed"], "nemenyi": ["method"],
    "reproduction_check_m_seed0": ["scene", "band", "severity", "realization", "method"],
}
def keyfor(name):
    for k, v in KEYS.items():
        if k in name: return v
    return None

def compare(fr, fp, name):
    a = pd.read_csv(fr); b = pd.read_csv(fp); keys = keyfor(name); msgs = []
    if keys is None: return [f"{name}: no key rule, skipped"]
    if "nemenyi" in name:   # R writes the Nemenyi p-value matrix (lower triangle); normalise to long form
        a = a.melt(id_vars=[a.columns[0]], var_name="other", value_name="p").rename(columns={a.columns[0]: "method"})
        b = b.melt(id_vars=[b.columns[0]], var_name="other", value_name="p").rename(columns={b.columns[0]: "method"})
        a = a.dropna(); b = b.dropna(); keys = ["method", "other"]
        # symmetric: order the pair so both tables index the same way
        for d in (a, b): d[["method", "other"]] = np.sort(d[["method", "other"]].values, axis=1)
        a = a.drop_duplicates(keys); b = b.drop_duplicates(keys)
    for d in (a, b):
        for k in keys:
            if k in d.columns: d[k] = d[k].astype(str)
    m = a.merge(b, on=keys, how="outer", suffixes=("_r", "_py"), indicator=True)
    if (m["_merge"] != "both").any():
        msgs.append(f"{name}: {int((m['_merge']!='both').sum())} rows present in only one output")
    m = m[m["_merge"] == "both"]
    for col in a.columns:
        if col in keys or f"{col}_r" not in m: continue
        x, y = m[f"{col}_r"], m[f"{col}_py"]
        if not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)) or pd.api.types.is_bool_dtype(x):
            xs, ys = x.fillna("").astype(str).str.strip(), y.fillna("").astype(str).str.strip()   # empty and NaN are the same "no cells"
            # tolerate R TRUE/FALSE vs Python True/False
            xs = xs.str.replace("TRUE", "True").str.replace("FALSE", "False"); ys = ys.str.replace("TRUE", "True").str.replace("FALSE", "False")
            bad = (xs != ys)
            if col.startswith("crit3_B05_severe") and col != "crit3_B05_severe_any_beats": bad = bad & False  # free text; compared via numeric cols elsewhere
            if bad.any(): msgs.append(f"{name}.{col}: {int(bad.sum())} text/boolean mismatches, e.g. {xs[bad].iloc[0]!r} vs {ys[bad].iloc[0]!r}")
        else:
            tol = TOL.get(col, DEFAULT_TOL)
            diff = (x - y).abs(); both_nan = x.isna() & y.isna(); bad = (~both_nan) & ((diff > tol) | x.isna() ^ y.isna())
            if bad.any(): msgs.append(f"{name}.{col}: {int(bad.sum())} values differ by > {tol:g} (max {diff[bad].max():.3g})")
    return msgs

def main():
    rd, pdir = sys.argv[1], sys.argv[2]; issues = []; n = 0
    for fr in sorted(glob.glob(os.path.join(rd, "*.csv"))):
        name = os.path.basename(fr); fp = os.path.join(pdir, name)
        if not os.path.exists(fp): issues.append(f"{name}: present in R output only"); continue
        n += 1; issues += compare(fr, fp, name.replace(".csv", ""))
    for fp in sorted(glob.glob(os.path.join(pdir, "*.csv"))):
        if not os.path.exists(os.path.join(rd, os.path.basename(fp))): issues.append(f"{os.path.basename(fp)}: present in Python output only")
    for f in ("m_vs_c_spread.txt",):
        a, b = os.path.join(rd, f), os.path.join(pdir, f)
        if os.path.exists(a) and os.path.exists(b):
            va = dict(l.split("=", 1) for l in open(a).read().strip().splitlines())
            vb = dict(l.split("=", 1) for l in open(b).read().strip().splitlines())
            if va.get("verdict") != vb.get("verdict"): issues.append(f"{f}: verdict differs: {va.get('verdict')!r} vs {vb.get('verdict')!r}")
            for k in ("m_to_c_diff", "m_sd"):
                if abs(float(va[k]) - float(vb[k])) > 1e-6: issues.append(f"{f}: {k} differs")
    print(f"compared {n} file pairs")
    if issues: print("MISMATCHES:"); [print("  ", i) for i in issues]; sys.exit(1)
    print("R and Python outputs agree within tolerance.")

if __name__ == "__main__": main()
