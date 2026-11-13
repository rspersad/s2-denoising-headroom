#!/usr/bin/env python3
# =============================================================================
# stats_sensitivity.py -- Section 3.8.7 detector-capacity sensitivity arm
#
# Python implementation of the Chapter 3 primary analysis, replicated per
# detector variant (m, c, e). Mirrors sensitivity_analysis.R file-for-file so
# the two can be diffed as the cross-implementation check.
#
#   * unit of analysis = matched block (scene x band x severity x realization)
#   * paired deltas of each denoising method vs the untreated degraded baseline,
#     per condition (band x severity) and pooled
#   * BCa bootstrap CI of the mean delta, 10,000 resamples (resampling blocks)
#   * Wilcoxon signed-rank per method, Benjamini-Hochberg within variant
#   * Friedman omnibus over the six treatments + Nemenyi post hoc, mean ranks
#   * per-variant, per-condition recoverable headroom H = D_clean - D_degraded
#   * Section 3.8.7 criteria 1-3, answered per variant
#   * seed spread (descriptive only); m-to-c difference vs the m seed SD
#   * optional reproduction check: m seed-0 rows against the Chapter 4 table
#
# NO inferential test across variants is performed.
#
# INPUT  : CSV with columns
#          variant, seed, scene, band, severity, realization, method, mAP50 [, mAP50_95]
#          Clean-reference rows carry method == CLEAN_LABEL. They may have
#          severity/realization populated (one per block) or blank (one per
#          scene x band); both are handled.
# USAGE  : python stats_sensitivity.py all_variants.csv out_dir [--primary ch4_table.csv]
# DEPS   : numpy, pandas, scipy >= 1.7
# =============================================================================
import argparse, sys, json, platform
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import studentized_range

# ----------------------------------------------------------------------------- CONFIG
BASELINE_LABEL    = "none"      # label in `method` for the untreated degraded baseline   <-- SET FROM TABLE
CLEAN_LABEL       = "clean"     # label in `method` for the clean reference                <-- SET FROM TABLE
SEVERE_LABEL      = "severe"    # label in `severity` for the severe level                 <-- SET FROM TABLE
B05_LABEL         = "B05"       # label in `band` for band 5                               <-- SET FROM TABLE
PRIMARY_METRIC    = "mAP50"
SECONDARY_METRIC  = "mAP50_95"
RESOLUTION        = 0.005       # measurement resolution, mAP@0.5: median within-cell SD of the untreated
                                # baseline across the three noise realizations, primary detector (0.0051)
N_BOOT            = 10000
BOOT_SEED         = 20261004
VARIANT_ORDER     = ["m", "c", "e"]
REFERENCE_VARIANT = "m"
SEED_FOR_INFERENCE = 0          # inference on the seed-0 detectors; seeds 1,2 feed spread only
BLOCK_KEYS        = ["scene", "band", "severity", "realization"]

# ----------------------------------------------------------------------------- HELPERS
def bca_mean(x, n_boot=N_BOOT, seed=BOOT_SEED, conf=0.95):
    """BCa interval for the mean of paired deltas. Blocks are the resampling unit."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 3 or np.std(x, ddof=1) == 0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    res = stats.bootstrap((x,), np.mean, n_resamples=n_boot, method="BCa",
                          confidence_level=conf, random_state=rng, vectorized=True)
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))

def add_tost(tbl, w):
    """[REV3] Equivalence against the resolution bound (pooled table only).
    tost_equivalent: the 90% BCa interval of the mean delta lies inside (-RESOLUTION, +RESOLUTION),
    which is the confidence-interval-inclusion form of two one-sided tests at alpha = 0.05
    (Schuirmann, 1987). tost_p: the paired-t form, the larger of the two one-sided p-values,
    reported for reference alongside the nonparametric verdict."""
    lo90, hi90, pt = [], [], []
    for m in tbl.method:
        x = w[f"delta_{m}"].to_numpy(float); x = x[np.isfinite(x)]
        l, h = bca_mean(x, conf=0.90); lo90.append(l); hi90.append(h)
        if x.size < 2 or np.std(x, ddof=1) == 0: pt.append(np.nan); continue
        p1 = stats.ttest_1samp(x, -RESOLUTION, alternative="greater").pvalue
        p2 = stats.ttest_1samp(x,  RESOLUTION, alternative="less").pvalue
        pt.append(float(max(p1, p2)))
    tbl["bca90_lo"] = lo90; tbl["bca90_hi"] = hi90; tbl["tost_p"] = pt
    tbl["tost_equivalent"] = (np.isfinite(tbl.bca90_lo) & (tbl.bca90_lo > -RESOLUTION) & (tbl.bca90_hi < RESOLUTION))
    return tbl

def bh(p):
    """Benjamini-Hochberg, NA-safe, same convention as R's p.adjust(method='BH')."""
    p = np.asarray(p, float); out = np.full_like(p, np.nan)
    ok = np.isfinite(p); pv = p[ok]; n = pv.size
    if n == 0: return out
    order = np.argsort(pv)[::-1]; ranked = pv[order]
    adj = np.minimum.accumulate(ranked * n / np.arange(n, 0, -1))
    adj = np.minimum(adj, 1.0); res = np.empty(n); res[order] = adj
    out[ok] = res; return out

def wilcoxon_p(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 2 or np.std(x, ddof=1) == 0: return np.nan
    try:   return float(stats.wilcoxon(x, zero_method="wilcox", correction=True,
                                       alternative="two-sided", mode="approx").pvalue)
    except Exception: return np.nan

def paired_wide(d, metric):
    w = d.pivot_table(index=BLOCK_KEYS, columns="method", values=metric, aggfunc="first")
    methods = [m for m in w.columns if m != BASELINE_LABEL]
    for m in methods: w[f"delta_{m}"] = w[m] - w[BASELINE_LABEL]
    return w.reset_index(), methods

def delta_table(w, methods):
    rows = []
    for m in methods:
        x = w[f"delta_{m}"].to_numpy(float); lo, hi = bca_mean(x)
        rows.append(dict(method=m, n_blocks=int(np.isfinite(x).sum()),
                         mean_delta=float(np.nanmean(x)), median_delta=float(np.nanmedian(x)),
                         bca_lo=lo, bca_hi=hi, wilcoxon_p=wilcoxon_p(x)))
    return pd.DataFrame(rows)

def friedman_block(d, metric):
    """Friedman over treatments with blocks as subjects; Nemenyi post hoc; mean ranks (1 = best)."""
    w = d.pivot_table(index=BLOCK_KEYS, columns="method", values=metric, aggfunc="first").dropna()
    M = w.to_numpy(float); k = M.shape[1]; n = M.shape[0]
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, M)
    mean_ranks = pd.Series(ranks.mean(0), index=w.columns).sort_values()
    # Nemenyi for Friedman: q_ij = |R_i - R_j| / sqrt(k(k+1)/(6n)); p from studentized range (k, inf)
    se = np.sqrt(k * (k + 1) / (6.0 * n)); names = list(w.columns)
    nem = pd.DataFrame(np.nan, index=names, columns=names)
    R = dict(zip(names, ranks.mean(0)))
    for i in names:
        for j in names:
            if i == j: continue
            q = abs(R[i] - R[j]) / se
            nem.loc[i, j] = float(studentized_range.sf(q * np.sqrt(2), k, np.inf))
    return dict(chi2=float(chi2), df=k - 1, p=float(p), n_blocks=n, mean_ranks=mean_ranks, nemenyi=nem)

def headroom_table(fact, clean, metric):
    """Per-condition H = D_clean - D_baseline. Clean rows match on block if they carry severity/
    realization, else on scene x band (broadcast across severity x realization)."""
    base = fact[fact.method == BASELINE_LABEL][BLOCK_KEYS + [metric]].rename(columns={metric: "D_degraded"})
    c = clean[["scene", "band", "severity", "realization", metric]].rename(columns={metric: "D_clean"})
    # Rule: one clean row per scene x band (the praxis design: 12 per detector, severity == "clean")
    # -> broadcast across severity x realization. More than one per scene x band -> match per block.
    one_per_sb = (c.groupby(["scene", "band"]).size() == 1).all()
    if one_per_sb: j = base.merge(c[["scene", "band", "D_clean"]], on=["scene", "band"], how="left")
    else:          j = base.merge(c, on=BLOCK_KEYS, how="left")
    if j["D_clean"].isna().any():
        raise SystemExit(f"headroom: {int(j['D_clean'].isna().sum())} baseline blocks have no clean-reference match; "
                         "check CLEAN_LABEL and the clean rows' scene/band values")
    j["headroom"] = j["D_clean"] - j["D_degraded"]
    per_cond = (j.groupby(["band", "severity"])
                 .agg(n_blocks=("headroom", "size"), D_clean=("D_clean", "mean"),
                      D_degraded=("D_degraded", "mean"), headroom=("headroom", "mean"),
                      headroom_sd=("headroom", "std")).reset_index())
    return j, per_cond

# ----------------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv"); ap.add_argument("out_dir")
    ap.add_argument("--primary", help="Chapter 4 primary table (m seed-0) for the reproduction check")
    ap.add_argument("--inference-seed", type=int, default=SEED_FOR_INFERENCE)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "run_log.txt", "w")
    def say(*s):
        t = " ".join(str(x) for x in s); print(t); log.write(t + "\n")

    dt = pd.read_csv(a.input_csv)
    dt = dt.rename(columns={"map50": "mAP50", "map50_95": "mAP50_95"})   # accept the sweep's lowercase names
    req = ["variant", "seed", "scene", "band", "severity", "realization", "method", PRIMARY_METRIC]
    miss = [c for c in req if c not in dt.columns]
    if miss: sys.exit(f"Input is missing columns: {miss}")
    dt["variant"] = pd.Categorical(dt["variant"].astype(str), categories=VARIANT_ORDER)
    say("Rows:", len(dt)); say("Rows per variant x seed:\n", dt.groupby(["variant", "seed"], observed=True).size().unstack())
    say("Methods:", ", ".join(sorted(dt["method"].astype(str).unique())))
    if BASELINE_LABEL not in set(dt["method"]): sys.exit(f"BASELINE_LABEL '{BASELINE_LABEL}' not found in method column")

    clean_all = dt[dt.method == CLEAN_LABEL].copy()
    fact_all  = dt[dt.method != CLEAN_LABEL].copy()
    inf = fact_all[fact_all.seed == a.inference_seed]
    if inf.empty: sys.exit(f"No rows for inference seed {a.inference_seed}")
    clean_inf = clean_all[clean_all.seed == a.inference_seed]

    results = {}
    for v in VARIANT_ORDER:
        d = inf[inf.variant == v]
        if d.empty: say(f"\n== Variant {v}: no rows, skipped =="); continue
        say(f"\n==================== Variant {v} (seed {a.inference_seed}) ====================")
        w, methods = paired_wide(d, PRIMARY_METRIC)

        pooled = delta_table(w, methods)
        pooled["p_bh"] = bh(pooled["wilcoxon_p"])
        pooled["beats_resolution"] = (pooled.mean_delta > RESOLUTION) & np.isfinite(pooled.bca_lo) & (pooled.bca_lo > 0)
        pooled = add_tost(pooled, w)                                   # [REV3] equivalence at +/- RESOLUTION
        pooled = pooled.sort_values("mean_delta", ascending=False).reset_index(drop=True)
        say(f"\nPooled paired deltas vs baseline ({PRIMARY_METRIC}), resolution = {RESOLUTION}:\n",
            pooled.round(4).to_string(index=False))
        pooled.to_csv(out / f"{v}_pooled_deltas_{PRIMARY_METRIC}.csv", index=False)

        pcs = []
        for (b, s), g in w.groupby(["band", "severity"]):
            t = delta_table(g, methods); t.insert(0, "severity", s); t.insert(0, "band", b); pcs.append(t)
        per_cond = pd.concat(pcs, ignore_index=True)
        per_cond["p_bh"] = bh(per_cond["wilcoxon_p"])   # BH across the whole per-condition family within variant
        per_cond["beats_resolution"] = (per_cond.mean_delta > RESOLUTION) & np.isfinite(per_cond.bca_lo) & (per_cond.bca_lo > 0)
        per_cond = per_cond.sort_values(["band", "severity", "mean_delta"], ascending=[True, True, False]).reset_index(drop=True)
        per_cond.to_csv(out / f"{v}_per_condition_deltas_{PRIMARY_METRIC}.csv", index=False)

        if SECONDARY_METRIC in d.columns:
            w2, m2 = paired_wide(d, SECONDARY_METRIC); p2 = delta_table(w2, m2); p2["p_bh"] = bh(p2["wilcoxon_p"])
            p2.to_csv(out / f"{v}_pooled_deltas_{SECONDARY_METRIC}.csv", index=False)

        fb = friedman_block(d, PRIMARY_METRIC)
        say(f"\nFriedman: chi2 = {fb['chi2']:.3f}, df = {fb['df']}, p = {fb['p']:.4g} (n blocks = {fb['n_blocks']})")
        say("Mean ranks (1 = best):\n", fb["mean_ranks"].round(3).to_string())
        say("Nemenyi post hoc p-values:\n", fb["nemenyi"].round(4).to_string())
        pd.DataFrame({"method": fb["mean_ranks"].index, "mean_rank": fb["mean_ranks"].values}).to_csv(out / f"{v}_mean_ranks.csv", index=False)
        fb["nemenyi"].to_csv(out / f"{v}_nemenyi.csv")

        hb, hc = (None, None)
        cv = clean_inf[clean_inf.variant == v]
        if not cv.empty:
            hb, hc = headroom_table(d, cv, PRIMARY_METRIC)
            say(f"\nRecoverable headroom per condition ({PRIMARY_METRIC}):\n", hc.round(4).to_string(index=False))
            hc.to_csv(out / f"{v}_headroom_per_condition.csv", index=False)
            hb.to_csv(out / f"{v}_headroom_per_block.csv", index=False)
        else:
            say("\n[no clean-reference rows for this variant: headroom not computed]")

        b05 = per_cond[(per_cond.band == B05_LABEL) & (per_cond.severity == SEVERE_LABEL)].reset_index(drop=True)
        if hc is not None and not b05.empty:
            h = hc[(hc.band == B05_LABEL) & (hc.severity == SEVERE_LABEL)]
            b05["headroom"] = float(h.headroom.iloc[0]) if not h.empty else np.nan
            b05["delta_over_headroom"] = b05.mean_delta / b05.headroom
        say("\nB05 severe cell:\n", b05.round(4).to_string(index=False) if not b05.empty else "cell not found")
        b05.to_csv(out / f"{v}_B05_severe.csv", index=False)
        results[v] = dict(pooled=pooled, per_cond=per_cond, friedman=fb, b05=b05, headroom=hc)

    # ------------------------------------------------------------------- criteria
    say("\n\n==================== Section 3.8.7 criteria ====================")
    ref_order = list(results[REFERENCE_VARIANT]["friedman"]["mean_ranks"].index)
    say(f"Reference ordering ({REFERENCE_VARIANT}): {' < '.join(ref_order)}")
    crit = []
    for v, r in results.items():
        c1 = r["pooled"][r["pooled"].beats_resolution].method.tolist()
        pc = r["per_cond"][r["per_cond"].beats_resolution]
        c1_cells = [f"{m}@{b}/{s}" for m, b, s in zip(pc.method, pc.band, pc.severity)]
        mr = list(r["friedman"]["mean_ranks"].index)
        b = r["b05"]
        c3 = (f"best method {b.method[0]}, delta {b.mean_delta[0]:.4f} [{b.bca_lo[0]:.4f}, {b.bca_hi[0]:.4f}], "
              f"BH p {b.p_bh[0]:.3g}, beats resolution {bool(b.beats_resolution[0])}"
              + (f", headroom {b.headroom[0]:.4f}" if "headroom" in b.columns else "")) if len(b) else "cell not found"
        c3_beats = bool(b.beats_resolution.any()) if len(b) else False
        crit.append(dict(variant=v,
                         crit1_pooled_any_beats=len(c1) > 0, crit1_pooled_methods=";".join(c1),
                         crit1_any_cell_beats=len(c1_cells) > 0, crit1_cells=";".join(c1_cells),
                         crit2_order_preserved=(mr == ref_order), crit2_baseline_best_rank=(mr[0] == BASELINE_LABEL),
                         crit2_order=" < ".join(mr),
                         crit3_B05_severe_any_beats=c3_beats, crit3_B05_severe=c3))
    crit = pd.DataFrame(crit); say(crit.to_string(index=False)); crit.to_csv(out / "criteria_3_8_7.csv", index=False)

    # ------------------------------------------------------------------- seed spread (descriptive)
    say("\n\n==================== Seed spread (descriptive, no test) ====================")
    sb = (fact_all[fact_all.method == BASELINE_LABEL].groupby(["variant", "seed"], observed=True)[PRIMARY_METRIC]
            .mean().rename("baseline_mAP").reset_index())
    if not clean_all.empty:
        sc = clean_all.groupby(["variant", "seed"], observed=True)[PRIMARY_METRIC].mean().rename("clean_mAP").reset_index()
        spread = sb.merge(sc, on=["variant", "seed"], how="outer")
    else: spread = sb
    agg = dict(n_seeds=("seed", "size"), baseline_mean=("baseline_mAP", "mean"), baseline_sd=("baseline_mAP", "std"))
    if "clean_mAP" in spread: agg.update(clean_mean=("clean_mAP", "mean"), clean_sd=("clean_mAP", "std"))
    summ = spread.groupby("variant", observed=True).agg(**agg).reset_index()
    say(summ.round(4).to_string(index=False))
    spread.to_csv(out / "seed_values.csv", index=False); summ.to_csv(out / "seed_spread.csv", index=False)
    if set(["m", "c"]) <= set(summ.variant.astype(str)):
        m_sd = float(summ.loc[summ.variant == "m", "baseline_sd"].iloc[0])
        diff = float(summ.loc[summ.variant == "c", "baseline_mean"].iloc[0] - summ.loc[summ.variant == "m", "baseline_mean"].iloc[0])
        direction = "c ABOVE m" if diff > 0 else "c BELOW m"
        verdict = ("m SD unavailable (fewer than 2 seeds)" if not np.isfinite(m_sd) else
                   f"INSIDE one m SD -> 'did not differ from' ({direction})" if abs(diff) <= m_sd
                   else f"OUTSIDE one m SD -> {direction}, |diff|/m_SD = {abs(diff)/m_sd:.2f}")
        say(f"\nm-to-c baseline difference = {diff:.4f}; m seed SD = {m_sd:.4f}; {verdict}")
        (out / "m_vs_c_spread.txt").write_text(f"m_to_c_diff={diff:.6f}\nm_sd={m_sd:.6f}\nverdict={verdict}\n")

    # hardware note: seed spread for m and c mixes 3090 and A4500 runs (handoff 1.4) -- state in text
    # ------------------------------------------------------------------- reproduction check
    if a.primary:
        say("\n\n==================== Reproduction check: m seed-0 vs Chapter 4 table ====================")
        prim = pd.read_csv(a.primary).rename(columns={"map50": "mAP50", "map50_95": "mAP50_95"})
        keys = BLOCK_KEYS + ["method"]
        m0 = dt[(dt.variant == "m") & (dt.seed == a.inference_seed)][keys + [PRIMARY_METRIC]]
        if "variant" in prim.columns: prim = prim[prim["variant"].astype(str) == "m"]
        if "seed" in prim.columns:    prim = prim[prim["seed"] == a.inference_seed]
        pk = prim[keys + [PRIMARY_METRIC]].drop_duplicates(keys).rename(columns={PRIMARY_METRIC: "primary"})
        j = m0.merge(pk, on=keys, how="inner"); j["abs_diff"] = (j[PRIMARY_METRIC] - j["primary"]).abs()
        say(f"matched rows: {len(j)} of {len(m0)}; max |diff| = {j.abs_diff.max():.6f}; "
            f"rows with |diff| > 1e-6: {(j.abs_diff > 1e-6).sum()}")
        j.to_csv(out / "reproduction_check_m_seed0.csv", index=False)

    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               scipy=__import__("scipy").__version__, host=platform.node(), n_boot=N_BOOT, boot_seed=BOOT_SEED,
               inference_seed=a.inference_seed, resolution=RESOLUTION,
               labels=dict(baseline=BASELINE_LABEL, clean=CLEAN_LABEL, severe=SEVERE_LABEL, b05=B05_LABEL))
    (out / "sessionInfo.json").write_text(json.dumps(env, indent=2))
    say("\nOutputs written to", out.resolve()); log.close()

if __name__ == "__main__": main()
