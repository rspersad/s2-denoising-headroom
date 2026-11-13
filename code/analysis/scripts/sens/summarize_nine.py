#!/usr/bin/env python3
# =============================================================================
# summarize_nine.py -- Chapter 4 tables for the nine-detector sensitivity arm
#
# Reads the assembled sweep table and the three stats_sensitivity.py output
# directories (seeds 0, 1, 2) and writes the summaries the Chapter 4 text cites.
# Everything here is descriptive post-processing of those outputs; no new
# inferential test is introduced.
#
# USAGE (Astraea):
#   python summarize_nine.py --table /data/praxis/results_sens/all_variants.csv \
#       --out-dirs out_py out_py_seed1 out_py_seed2 --out /data/praxis/results_sens/summary
#
# OUTPUTS (in --out):
#   nine_detector_summary.csv   one row per variant x seed: BM3D and Restormer pooled
#                               deltas with 95% BCa and TOST verdicts, cells beating
#                               resolution, criterion 2, B05/severe best method, headroom
#   ceiling_all_contrasts.csv   405 rows (9 detectors x 9 cells x 5 methods): per-cell gain,
#                               headroom, exceedance = gain - headroom with its SE, dPSNR, dSSIM
#   ceiling_exceedances.csv     the subset with exceedance > 0, with z = exceedance / SE
#   threshold_sensitivity.csv   count of pooled and per-cell contrasts passing the recovery
#                               rule (mean > t AND BCa lower bound > 0) at t = 0.002 ... 0.008
#   resolution_basis.csv        within-cell replicate SD of untreated-baseline mAP@0.5 per
#                               variant x seed (median, mean, max over the 36 cells)
#   headroom_gain_spearman.csv  Spearman rho between per-cell headroom and per-cell gain,
#                               per method, per variant x seed (9 cells) and across all nine
#                               detectors (81 cells)
# =============================================================================
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

BASELINE, CLEAN = "none", "clean"
BENEFICIAL = ["bm3d", "restormer"]
CELL = ["band", "severity"]
BLOCK = ["scene", "band", "severity", "realization"]

def load_table(p):
    d = pd.read_csv(p).rename(columns={"map50": "mAP50", "map50_95": "mAP50_95", "PSNR": "psnr", "SSIM": "ssim"})
    for c in ("psnr", "ssim"):
        if c in d.columns: d[c] = pd.to_numeric(d[c].replace("inf", np.inf), errors="coerce")
    d["variant"] = d["variant"].astype(str); d["seed"] = d["seed"].astype(int)
    return d

def seed_of(outdir):
    si = Path(outdir) / "sessionInfo.json"
    if si.exists(): return int(json.load(open(si))["inference_seed"]), json.load(open(si)).get("resolution")
    sys.exit(f"{outdir}: no sessionInfo.json; is this a stats_sensitivity.py output directory?")

def per_block_frames(d):
    """Per detector: wide block table with D_none, D_clean (broadcast per scene x band), D_<method>."""
    fact = d[d.method != CLEAN]; clean = d[d.method == CLEAN]
    w = fact.pivot_table(index=BLOCK, columns="method", values="mAP50", aggfunc="first").reset_index()
    c = clean.groupby(["scene", "band"])["mAP50"].first().rename("D_clean").reset_index()
    w = w.merge(c, on=["scene", "band"], how="left")
    if w.D_clean.isna().any(): sys.exit("clean-reference match failed; check scene/band labels")
    # perceptual columns are optional: the assembled sweep table may carry only the detection metrics
    piv = lambda col: (fact.pivot_table(index=BLOCK, columns="method", values=col, aggfunc="first").reset_index()
                       if col in fact.columns and fact[col].notna().any() else None)
    return w, piv("psnr"), piv("ssim")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True); ap.add_argument("--out-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = load_table(a.table)
    dirs = {}
    res_vals = set()
    for od in a.out_dirs:
        s, r = seed_of(od); dirs[s] = Path(od); res_vals.add(r)
    if len(res_vals) > 1: sys.exit(f"output directories were produced at different RESOLUTION values: {res_vals}")
    RES = res_vals.pop()
    print(f"resolution in outputs: {RES}; seeds: {sorted(dirs)}")
    for col in ("psnr", "ssim"):
        if col not in d.columns or d[col].isna().all(): print(f"note: table has no {col} column; d{col} will be NaN in the ceiling tables")
    variants = [v for v in ["m", "c", "e"] if v in set(d.variant)]
    methods = sorted(m for m in d.method.unique() if m not in (BASELINE, CLEAN))

    summary, contrasts, resbasis, cell_gain = [], [], [], []
    for v in variants:
        for s in sorted(dirs):
            dd = d[(d.variant == v) & (d.seed == s)]
            if dd.empty: continue
            od = dirs[s]
            pooled = pd.read_csv(od / f"{v}_pooled_deltas_mAP50.csv").set_index("method")
            percond = pd.read_csv(od / f"{v}_per_condition_deltas_mAP50.csv")
            crit = pd.read_csv(od / "criteria_3_8_7.csv").set_index("variant").loc[v]
            hc = pd.read_csv(od / f"{v}_headroom_per_condition.csv")
            b05 = pd.read_csv(od / f"{v}_B05_severe.csv")
            w, ps, ss = per_block_frames(dd)

            row = dict(variant=v, seed=s,
                       baseline_mean_mAP50=float(w[BASELINE].mean()), clean_mean_mAP50=float(w.D_clean.mean()),
                       pooled_headroom=float((w.D_clean - w[BASELINE]).mean()))
            for m in BENEFICIAL:
                if m in pooled.index:
                    r = pooled.loc[m]
                    row.update({f"{m}_delta": r.mean_delta, f"{m}_lo95": r.bca_lo, f"{m}_hi95": r.bca_hi,
                                f"{m}_p_bh": r.p_bh, f"{m}_beats_resolution": bool(r.beats_resolution)})
                    if "tost_equivalent" in pooled.columns:
                        row.update({f"{m}_lo90": r.bca90_lo, f"{m}_hi90": r.bca90_hi,
                                    f"{m}_tost_equivalent": bool(r.tost_equivalent), f"{m}_tost_p": r.tost_p})
            pc_beats = percond[percond.beats_resolution.astype(str).str.lower() == "true"]
            row.update(cells_beating_resolution=int(len(pc_beats)),
                       cells_beating=";".join(f"{m}@{b}/{sv}" for m, b, sv in zip(pc_beats.method, pc_beats.band, pc_beats.severity)),
                       crit1_pooled_methods=str(crit.crit1_pooled_methods) if pd.notna(crit.crit1_pooled_methods) else "",
                       crit2_baseline_best_rank=bool(crit.crit2_baseline_best_rank),
                       crit2_order_preserved=bool(crit.crit2_order_preserved), crit2_order=crit.crit2_order)
            if len(b05):
                b = b05.iloc[0]
                row.update(b05sev_best=b.method, b05sev_delta=b.mean_delta, b05sev_lo95=b.bca_lo, b05sev_hi95=b.bca_hi,
                           b05sev_beats=bool(b.beats_resolution), b05sev_headroom=b.get("headroom", np.nan),
                           b05sev_delta_over_headroom=b.get("delta_over_headroom", np.nan))
            summary.append(row)

            # replicate SD of the untreated baseline within scene x band x severity
            rep = w.groupby(["scene", "band", "severity"])[BASELINE].std(ddof=1)
            resbasis.append(dict(variant=v, seed=s, n_cells=int(rep.size), replicate_sd_median=rep.median(),
                                 replicate_sd_mean=rep.mean(), replicate_sd_max=rep.max()))

            # ceiling contrasts: per cell, gain vs headroom; exceedance = mean(D_m - D_clean) over blocks
            for (b, sv), g in w.groupby(CELL):
                gp = ps[(ps.band == b) & (ps.severity == sv)] if ps is not None else pd.DataFrame()
                gs = ss[(ss.band == b) & (ss.severity == sv)] if ss is not None else pd.DataFrame()
                H = (g.D_clean - g[BASELINE])
                for m in methods:
                    if m not in g.columns: continue
                    gain = g[m] - g[BASELINE]; exc = g[m] - g.D_clean
                    se = float(exc.std(ddof=1) / np.sqrt(exc.notna().sum())) if exc.notna().sum() > 1 else np.nan
                    contrasts.append(dict(variant=v, seed=s, band=b, severity=sv, method=m, n_blocks=int(gain.notna().sum()),
                                          gain=float(gain.mean()), headroom=float(H.mean()), exceedance=float(exc.mean()),
                                          exceedance_se=se, z=float(exc.mean() / se) if se and np.isfinite(se) and se > 0 else np.nan,
                                          dpsnr=float((gp[m] - gp[BASELINE]).mean()) if m in gp.columns else np.nan,
                                          dssim=float((gs[m] - gs[BASELINE]).mean()) if m in gs.columns else np.nan))
                    cell_gain.append(dict(variant=v, seed=s, band=b, severity=sv, method=m,
                                          gain=float(gain.mean()), headroom=float(H.mean())))

    summary = pd.DataFrame(summary); summary.to_csv(out / "nine_detector_summary.csv", index=False)
    con = pd.DataFrame(contrasts)
    # one-sided test of mean(D_method - D_clean) > 0 on n-1 df, BH across all contrasts (one family)
    con["p_one_sided"] = stats.t.sf(con.z, df=con.n_blocks - 1)
    ok = con.p_one_sided.notna(); ranks = con.loc[ok, "p_one_sided"].rank(method="first"); m_ = int(ok.sum())
    adj = (con.loc[ok, "p_one_sided"] * m_ / ranks).clip(upper=1.0)
    order = con.loc[ok, "p_one_sided"].sort_values(ascending=False).index
    adj.loc[order] = adj.loc[order].cummin(); con.loc[ok, "p_bh"] = adj
    con["positive_gain"] = con.gain > 0
    con.to_csv(out / "ceiling_all_contrasts.csv", index=False)
    exc = con[con.exceedance > 0].sort_values("z", ascending=False); exc.to_csv(out / "ceiling_exceedances.csv", index=False)
    pd.DataFrame(resbasis).to_csv(out / "resolution_basis.csv", index=False)

    # threshold sensitivity of the recovery rule, pooled (per detector) and per cell
    rows = []
    for t in [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]:
        n_pool_ben = n_pool_all = n_cell_ben = n_cell_all = 0
        for v in variants:
            for s in sorted(dirs):
                od = dirs[s]
                p = pd.read_csv(od / f"{v}_pooled_deltas_mAP50.csv"); pc = pd.read_csv(od / f"{v}_per_condition_deltas_mAP50.csv")
                hit = lambda df: (df.mean_delta > t) & np.isfinite(df.bca_lo) & (df.bca_lo > 0)
                n_pool_all += int(hit(p).sum()); n_pool_ben += int(hit(p[p.method.isin(BENEFICIAL)]).sum())
                n_cell_all += int(hit(pc).sum()); n_cell_ben += int(hit(pc[pc.method.isin(BENEFICIAL)]).sum())
        rows.append(dict(threshold=t, pooled_bm3d_restormer=n_pool_ben, pooled_all_methods=n_pool_all,
                         cells_bm3d_restormer=n_cell_ben, cells_all_methods=n_cell_all,
                         marker="<-- RESOLUTION" if abs(t - RES) < 1e-9 else ""))
    ts = pd.DataFrame(rows); ts.to_csv(out / "threshold_sensitivity.csv", index=False)

    # Spearman(headroom, gain) across cells
    cg = pd.DataFrame(cell_gain); sp = []
    for m in methods:
        g = cg[cg.method == m]
        for (v, s), gg in g.groupby(["variant", "seed"]):
            rho, p = stats.spearmanr(gg.headroom, gg.gain)
            sp.append(dict(method=m, scope=f"{v} seed {s}", n_cells=len(gg), rho=rho, p=p))
        rho, p = stats.spearmanr(g.headroom, g.gain)
        sp.append(dict(method=m, scope="all nine detectors", n_cells=len(g), rho=rho, p=p))
    order = {f"{v} seed {s}": i for i, (v, s) in enumerate((v, s) for v in variants for s in sorted(dirs))}
    sp.sort(key=lambda r: (methods.index(r["method"]), order.get(r["scope"], 99)))
    pd.DataFrame(sp).to_csv(out / "headroom_gain_spearman.csv", index=False)

    # console report
    pd.set_option("display.width", 250)
    cols = ["variant", "seed", "bm3d_delta", "bm3d_lo95", "bm3d_hi95", "restormer_delta", "restormer_lo95", "restormer_hi95",
            "cells_beating_resolution", "crit2_baseline_best_rank", "b05sev_best", "b05sev_delta", "b05sev_beats", "b05sev_headroom"]
    if "bm3d_tost_equivalent" in summary: cols[5:5] = ["bm3d_tost_equivalent"]; cols.insert(cols.index("cells_beating_resolution"), "restormer_tost_equivalent")
    print("\n== nine-detector summary (mAP@0.5) =="); print(summary[[c for c in cols if c in summary]].round(4).to_string(index=False))
    pos = exc[exc.positive_gain]
    print(f"\n== ceiling: {len(con)} contrasts; {len(exc)} with gain > headroom ({int((exc.z <= 2).sum())} within 2 SE, {int((exc.z > 2).sum())} beyond); "
          f"{len(pos)} of those with a positive gain ({int((pos.z <= 2).sum())} within 2 SE, {int((pos.z > 2).sum())} beyond); "
          f"{int((exc.p_bh < 0.05).sum())} survive BH at q = 0.05 across all {len(con)} contrasts ==")
    print(f"expected beyond 2 SE by chance if no contrast exceeds: about {0.023 * len(con):.0f}")
    print(exc[exc.z > 2][["variant", "seed", "band", "severity", "method", "gain", "headroom", "exceedance", "exceedance_se", "z", "p_one_sided", "p_bh", "dpsnr"]].round(4).to_string(index=False))
    print("\n== threshold sensitivity of the recovery rule =="); print(ts.to_string(index=False))
    print("\n== resolution basis: replicate SD of untreated baseline =="); print(pd.DataFrame(resbasis).round(4).to_string(index=False))
    print("\n== Spearman(headroom, gain), beneficial methods ==")
    spd = pd.DataFrame(sp); print(spd[spd.method.isin(BENEFICIAL)].round(4).to_string(index=False))
    print("\nwritten to", out.resolve())

if __name__ == "__main__": main()
