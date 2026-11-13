#!/usr/bin/env python3
"""
stats_factorial.py - Section 3.8 statistical analysis of the denoising factorial.

Tests the perceptual-vs-detection decoupling hypothesis on factorial_results.csv:
  - paired per-condition deltas (each denoising method vs the 'none' baseline),
  - BCa bootstrap 95% CIs on the mean delta for both axes (mAP50 and PSNR),
  - Wilcoxon signed-rank paired tests, Benjamini-Hochberg corrected across methods,
  - Friedman omnibus test across all treatments with a Nemenyi critical-difference
    post-hoc (computed for mAP50; the task axis).

The headline test of the thesis is: DeltaPSNR significantly > 0 (denoising improves
perceptual quality) while DeltamAP50 CIs straddle 0 (no detection effect) -- i.e. the
two axes are decoupled.

Runs on whatever methods are present, so it is meaningful at the BM3D stage and
again once the learned methods are scored. Requires numpy + scipy.

    python3 scripts/stats_factorial.py --results /data/praxis/results/factorial_results.csv \
        --out /data/praxis/results
"""
import argparse, csv, json, os, collections
import numpy as np
from scipy import stats

BLOCK_KEYS = ("band", "severity", "scene", "realization")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(path):
    return list(csv.DictReader(open(path)))


def paired_deltas(rows, method, baseline, metric):
    """Per-condition (method - baseline) deltas over matched blocks."""
    base = {tuple(r[k] for k in BLOCK_KEYS): fnum(r[metric]) for r in rows if r["method"] == baseline}
    out = []
    for r in rows:
        if r["method"] != method:
            continue
        key = tuple(r[k] for k in BLOCK_KEYS)
        if key in base:
            d = fnum(r[metric]) - base[key]
            if np.isfinite(d):
                out.append(d)
    return np.asarray(out, float)


def bca_ci(x, n_boot=10000, alpha=0.05, seed=0):
    """Bias-corrected & accelerated bootstrap CI for the mean."""
    x = np.asarray(x, float); n = x.size
    if n < 2:
        m = float(x.mean()) if n else float("nan")
        return m, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    theta = x.mean()
    boot = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    z0 = stats.norm.ppf((np.sum(boot < theta) + 0.5) / (n_boot + 1))
    jack = (x.sum() - x) / (n - 1)                      # leave-one-out means
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    adj = lambda z: stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo = np.quantile(boot, np.clip(adj(zl), 0, 1))
    hi = np.quantile(boot, np.clip(adj(zu), 0, 1))
    return float(theta), float(lo), float(hi)


def benjamini_hochberg(pvals, q=0.05):
    p = np.asarray(pvals, float); m = p.size
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * q
    passed = p[order] <= thresh
    crit = np.where(passed)[0]
    cutoff = p[order][crit.max()] if crit.size else -1.0
    reject = p <= cutoff
    # BH-adjusted p-values
    padj = np.empty(m); run = 1.0
    for i in range(m - 1, -1, -1):
        run = min(run, p[order][i] * m / (i + 1))
        padj[order[i]] = run
    return reject, padj


def friedman_nemenyi(rows, treatments, metric, alpha=0.05):
    """Friedman omnibus + Nemenyi CD over complete blocks."""
    by_block = collections.defaultdict(dict)
    for r in rows:
        if r["method"] in treatments:
            by_block[tuple(r[k] for k in BLOCK_KEYS)][r["method"]] = fnum(r[metric])
    blocks = [b for b, d in by_block.items() if all(t in d and np.isfinite(d[t]) for t in treatments)]
    k, N = len(treatments), len(blocks)
    if k < 3 or N < 2:
        return {"ok": False, "reason": f"need >=3 treatments and >=2 complete blocks (k={k}, N={N})"}
    M = np.array([[by_block[b][t] for t in treatments] for b in blocks])  # N x k
    chi, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    ranks = np.array([stats.rankdata(row) for row in M])  # higher metric = higher rank
    mean_ranks = ranks.mean(axis=0)
    try:
        q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
    except Exception:
        q = float("nan")
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N)) if np.isfinite(q) else float("nan")
    return {"ok": True, "chi2": float(chi), "p": float(p), "N": N, "k": k,
            "treatments": list(treatments),
            "mean_ranks": {t: float(mr) for t, mr in zip(treatments, mean_ranks)},
            "rank_direction": "ascending: higher mean rank is better",   # [NEW]
            "CD": float(cd)}


def coverage_split(rows, treatments):                                   # [NEW]
    """Split treatments into those observed in every block of the full design
    and those with restricted coverage. The 'clean' reference is excluded: it
    has no severity level and forms no comparable block."""
    blocks, by_t = set(), collections.defaultdict(set)
    for r in rows:
        if r["method"] == "clean":
            continue
        key = tuple(r[k] for k in BLOCK_KEYS)
        blocks.add(key)
        by_t[r["method"]].add(key)
    full = [t for t in treatments if len(by_t[t]) == len(blocks)]
    short = [t for t in treatments if t not in full]
    return full, short, len(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="none")
    args = ap.parse_args()

    rows = load(args.results)
    present = [m for m in dict.fromkeys(r["method"] for r in rows)]
    denoisers = [m for m in present if m not in ("clean", args.baseline)]
    print(f"methods present: {present}")
    print(f"baseline: {args.baseline} | denoisers tested: {denoisers}\n")

    report = {"baseline": args.baseline, "per_method": {}, "friedman": {}}

    # ---- per-method paired deltas, BCa CIs, Wilcoxon (both axes) ----
    pvals, labels = [], []
    print(f"{'method':14s}{'axis':6s}{'mean d':>10s}{'95% BCa CI':>22s}{'wilcoxon p':>12s}")
    for m in denoisers:
        report["per_method"][m] = {}
        for axis, metric in (("mAP50", "map50"), ("PSNR", "psnr")):
            d = paired_deltas(rows, m, args.baseline, metric)
            if d.size == 0:
                continue
            mean, lo, hi = bca_ci(d)
            try:
                _, pw = stats.wilcoxon(d) if np.any(d != 0) else (np.nan, 1.0)
            except ValueError:
                pw = 1.0
            report["per_method"][m][axis] = {"n": int(d.size), "mean": mean,
                                             "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": float(pw)}
            pvals.append(pw); labels.append((m, axis))
            print(f"{m:14s}{axis:6s}{mean:10.4f}   [{lo:8.4f},{hi:8.4f}]{pw:12.3g}")

    # ---- Benjamini-Hochberg across all paired tests ----
    if pvals:
        reject, padj = benjamini_hochberg(pvals, q=0.05)
        print("\n=== Benjamini-Hochberg (q=0.05) ===")
        for (m, axis), pr, rj in zip(labels, padj, reject):
            report["per_method"][m][axis]["bh_padj"] = float(pr)
            report["per_method"][m][axis]["bh_reject"] = bool(rj)
            print(f"  {m:14s}{axis:6s} p_adj={pr:.3g}  {'SIGNIFICANT' if rj else 'n.s.'}")

    # ---- Friedman + Nemenyi across all treatments (task axis) ----
    treatments = [args.baseline] + denoisers
    for axis, metric in (("mAP50", "map50"), ("PSNR", "psnr")):
        fn = friedman_nemenyi(rows, treatments, metric)
        report["friedman"][axis] = fn
        print(f"\n=== Friedman ({axis}) ===")
        if not fn["ok"]:
            print(f"  skipped: {fn['reason']}")
            continue
        print(f"  chi2={fn['chi2']:.2f}  p={fn['p']:.3g}  (N={fn['N']} blocks, k={fn['k']})")
        print(f"  CD={fn['CD']:.3f} | mean ranks: " +
              ", ".join(f"{t}={fn['mean_ranks'][t]:.2f}" for t in treatments))

    # ---- [NEW] Friedman restricted to full-severity-coverage treatments ----
    # The omnibus above requires complete blocks across all six treatments,
    # which restricts it to nominal severity because DnCNN and Restormer were
    # evaluated at nominal only. This second pass drops those two and tests the
    # same null across all three severity levels.
    full_cov, short_cov, n_blocks = coverage_split(rows, treatments)
    report["coverage"] = {"n_blocks_full_design": n_blocks,
                          "full_coverage_treatments": full_cov,
                          "restricted_coverage_treatments": short_cov}
    report["friedman_all_severity"] = {}
    print(f"\n=== coverage: {n_blocks} blocks in the full design ===")
    print(f"  full coverage: {', '.join(full_cov)}")
    print(f"  restricted:    {', '.join(short_cov) if short_cov else '(none)'}")
    if len(full_cov) >= 3:
        for axis, metric in (("mAP50", "map50"), ("PSNR", "psnr")):
            fn = friedman_nemenyi(rows, full_cov, metric)
            report["friedman_all_severity"][axis] = fn
            print(f"\n=== Friedman, all severities ({axis}) ===")
            if not fn["ok"]:
                print(f"  skipped: {fn['reason']}")
                continue
            print(f"  chi2={fn['chi2']:.2f}  p={fn['p']:.3g}  (N={fn['N']} blocks, k={fn['k']})")
            print(f"  CD={fn['CD']:.3f} | mean ranks: " +
                  ", ".join(f"{t}={fn['mean_ranks'][t]:.2f}" for t in full_cov))
    else:
        print("  fewer than 3 full-coverage treatments; all-severity omnibus not run")

    out_json = os.path.join(args.out, "stats_summary.json")
    json.dump(report, open(out_json, "w"), indent=2)
    print(f"\n[written] {out_json}")


if __name__ == "__main__":
    main()
