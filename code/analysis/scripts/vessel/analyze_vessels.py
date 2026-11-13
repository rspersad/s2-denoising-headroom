"""Vessel-level test of H2 (band detectability) from vessels.csv produced by score_vessels.py.
Usage: python3 -I analyze_vessels.py --vessels vessels.csv --out DIR [--thr 0.25]

Per condition (clean reference; untreated at each severity, realizations averaged):
  * matched confidence per vessel x band: Friedman across bands, Page's L for the ordered alternative B04 > B05 > B11,
    Wilcoxon pairwise with BH;
  * detected (conf >= thr) per vessel x band: Cochran's Q, McNemar pairwise with BH; also at thr 0.10 and 0.50;
  * per-scene versions of the same, and detection rate by band x vessel-size tercile.
Then a GEE logistic regression (statsmodels, if installed) of detected on band x log(area) + severity with scene clusters.
Writes CSVs for R (glmer) as well.
"""
import argparse, os, itertools, numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, page_trend_test, wilcoxon, false_discovery_control, chi2, binomtest

ap = argparse.ArgumentParser(); ap.add_argument("--vessels", required=True); ap.add_argument("--out", required=True); ap.add_argument("--thr", type=float, default=0.25)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
BANDS = ["B04", "B05", "B11"]
d = pd.read_csv(a.vessels)
d = d[d.method.isin(["clean", "none"])]
d["area"] = d.w * d.h

def cochran_q(X):  # X: n x k binary
    X = np.asarray(X, int); k = X.shape[1]; rs = X.sum(1); cs = X.sum(0); N = X.sum()
    keep = (rs > 0) & (rs < k); X = X[keep]; rs = rs[keep]
    if len(X) == 0: return np.nan, np.nan, 0
    Q = (k - 1) * (k * (cs ** 2).sum() - N ** 2) / (k * N - (rs ** 2).sum())
    return Q, chi2.sf(Q, k - 1), len(X)

def mcnemar(x, y):
    b = int(((x == 1) & (y == 0)).sum()); c = int(((x == 0) & (y == 1)).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0)

out = []
def analyse(label, P):
    """P: vessel x band pivot of confidence (index vessel_id), with scene column."""
    C = P[BANDS].values
    res = dict(condition=label, n_vessels=len(P))
    res["conf_mean_B04"], res["conf_mean_B05"], res["conf_mean_B11"] = C.mean(0).round(4)
    res["friedman_chi2"], res["friedman_p"] = friedmanchisquare(*C.T)
    pg = page_trend_test(C[:, ::-1], predicted_ranks=[1, 2, 3], method="asymptotic")  # B11 < B05 < B04
    res["page_L"], res["page_p"] = pg.statistic, pg.pvalue
    ps = []
    for x, y in (("B04", "B05"), ("B05", "B11"), ("B04", "B11")):
        diff = P[x] - P[y]; p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
        res[f"wilcoxon_{x}-{y}_mean"] = diff.mean().round(4); ps.append(p)
    for (x, y), pb in zip((("B04", "B05"), ("B05", "B11"), ("B04", "B11")), false_discovery_control(ps, method="bh")): res[f"wilcoxon_{x}-{y}_pBH"] = pb
    for thr in sorted({a.thr, 0.10, 0.50}):
        D = (C >= thr).astype(int)
        q, p, n = cochran_q(D); res[f"det_rate_{thr}"] = "/".join(f"{v:.3f}" for v in D.mean(0)); res[f"cochranQ_{thr}"] = q; res[f"cochranQ_p_{thr}"] = p; res[f"discordant_{thr}"] = n
        ps = []; bc = []
        for i, j in ((0, 1), (1, 2), (0, 2)):
            b, c, p = mcnemar(D[:, i], D[:, j]); bc.append(f"{b}:{c}"); ps.append(p)
        for (i, j), s, pb in zip(((0, 1), (1, 2), (0, 2)), bc, false_discovery_control(ps, method="bh")): res[f"mcnemar_{BANDS[i]}-{BANDS[j]}_{thr}"] = f"{s} p={pb:.3g}"
    out.append(res)

def pivot(sub):
    g = sub.groupby(["vessel_id", "band"])["conf"].mean().unstack()
    g = g.dropna(subset=BANDS)  # vessel must be present in all three bands
    g["scene"] = [v.split("_")[0] for v in g.index]; g["area"] = sub.groupby("vessel_id")["area"].first().reindex(g.index)
    return g

conds = [("clean", d[d.method == "clean"])] + [(f"untreated {s}", d[(d.method == "none") & (d.severity == s)]) for s in ("nominal", "moderate", "severe")]
conds.append(("untreated all severities", d[d.method == "none"]))
for label, sub in conds:
    if len(sub) == 0: continue
    P = pivot(sub); analyse(label, P)
    for sc, Ps in P.groupby("scene"): analyse(f"{label} | {sc}", Ps)
    # size terciles
    P["size"] = pd.qcut(P.area, 3, labels=["small", "medium", "large"])
    for sz, Ps in P.groupby("size", observed=True): analyse(f"{label} | {sz} (area {Ps.area.min():.0f}-{Ps.area.max():.0f} px)", Ps)
R = pd.DataFrame(out); R.to_csv(os.path.join(a.out, "vessel_band_tests.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(R[["condition", "n_vessels", "conf_mean_B04", "conf_mean_B05", "conf_mean_B11", "friedman_p", "page_p", f"det_rate_{a.thr}", f"cochranQ_p_{a.thr}"]].to_string(index=False))

# long table for GEE / glmer
L = d[d.method == "none"].groupby(["vessel_id", "scene", "band", "severity"], as_index=False).agg(conf=("conf", "mean"), area=("area", "first"))
L["detected"] = (L.conf >= a.thr).astype(int); L["log_area"] = np.log(L.area)
L.to_csv(os.path.join(a.out, "vessel_long_untreated.csv"), index=False)
try:
    import statsmodels.formula.api as smf, statsmodels.api as sm
    m = smf.gee("detected ~ C(band, Treatment('B04')) * log_area + C(severity)", groups="scene", data=L, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    open(os.path.join(a.out, "gee_summary.txt"), "w").write(m.summary().as_text()); print(m.summary())
except ImportError:
    print("statsmodels not installed: fit in R with lme4::glmer(detected ~ band * log_area + severity + (1|scene) + (1|vessel_id), family=binomial) on vessel_long_untreated.csv")
