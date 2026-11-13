#!/usr/bin/env Rscript
# =============================================================================
# sensitivity_analysis.R -- Section 3.8.7 detector-capacity sensitivity arm
# Revision 3, 2026-10-04. Changes from revision 1 are marked [REV2], from revision 2 [REV3].
#   [REV3] RESOLUTION = 0.005 (derived, see CONFIG); TOST equivalence columns on the pooled table.
#
# Replicates the Chapter 3 primary analysis per detector variant (m, c, e):
#   * unit of analysis = matched block (scene x band x severity x realization)
#   * paired deltas of each denoising method vs the untreated degraded baseline,
#     per condition (band x severity) and pooled
#   * BCa bootstrap CI of the mean delta, 10,000 resamples (resampling blocks)
#   * Wilcoxon signed-rank per method, Benjamini-Hochberg within variant
#   * Friedman omnibus over the six treatments (blocks as subjects) + Nemenyi
#     post hoc, with mean ranks
#   * [REV2] per-variant, per-condition recoverable headroom H = D_clean - D_degraded
#   * Section 3.8.7 criteria 1-3 per variant; [REV2] criterion 1 is reported
#     both pooled and per cell, since a capacity-dependent effect confined to
#     one cell is invisible to the pooled test
#   * seed spread (descriptive only); m-to-c difference vs the m seed SD
#   * [REV2] optional reproduction check: m seed-0 rows vs the Chapter 4 table
#
# NO inferential test across variants is performed (not a stated hypothesis).
#
# INPUT: CSV with columns
#   variant, seed, scene, band, severity, realization, method, mAP50 [, mAP50_95]
#   Clean-reference rows carry method == CLEAN_LABEL and may have
#   severity/realization populated (one per block) or empty (one per
#   scene x band); both are handled.
#
# USAGE:
#   Rscript sensitivity_analysis.R all_variants.csv out_dir/ [chapter4_primary_table.csv]
#
# Output file names are identical to stats_sensitivity.py so the two runs can
# be diffed directory against directory.
#
# Dependencies: data.table, boot, PMCMRplus.
# =============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(boot)
  library(PMCMRplus)
})

# ----------------------------------------------------------------------------- CONFIG
args        <- commandArgs(trailingOnly = TRUE)
INPUT_CSV   <- if (length(args) >= 1) args[1] else "results_sens/all_variants.csv"
OUT_DIR     <- if (length(args) >= 2) args[2] else "sensitivity_out"
PRIMARY_TBL <- if (length(args) >= 3) args[3] else NA_character_     # [REV2] Chapter 4 table for the reproduction check

BASELINE_LABEL   <- "none"        # label in `method` for the untreated degraded baseline  <-- SET FROM TABLE
CLEAN_LABEL      <- "clean"       # label in `method` for the clean reference               <-- SET FROM TABLE
SEVERE_LABEL     <- "severe"      # label in `severity` for the severe level               <-- SET FROM TABLE
B05_LABEL        <- "B05"         # label in `band` for band 5                              <-- SET FROM TABLE
PRIMARY_METRIC   <- "mAP50"
SECONDARY_METRIC <- "mAP50_95"
RESOLUTION       <- 0.005         # measurement resolution, mAP@0.5: median within-cell SD of the untreated
                                  # baseline across the three noise realizations, primary detector (0.0051)
N_BOOT           <- 10000
BOOT_SEED        <- 20261004
VARIANT_ORDER    <- c("m", "c", "e")
REFERENCE_VARIANT <- "m"
SEED_FOR_INFERENCE <- 0           # inference on the seed-0 detectors; seeds 1 and 2 feed spread only.
                                  # [REV2] Pooling seeds as extra blocks is NOT supported: blocks would
                                  # repeat across seeds and dcast would aggregate silently.
if (is.na(SEED_FOR_INFERENCE)) stop("SEED_FOR_INFERENCE must be a single seed; pooling seeds is not the design.")
BLOCK_KEYS <- c("scene", "band", "severity", "realization")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
set.seed(BOOT_SEED)
log_con <- file(file.path(OUT_DIR, "run_log.txt"), open = "wt")
say <- function(...) { msg <- paste0(..., collapse = ""); cat(msg, "\n"); writeLines(msg, log_con) }
say_tbl <- function(x, digits = 4) { s <- capture.output(print(x, digits = digits)); cat(s, sep = "\n"); writeLines(s, log_con) }

# ----------------------------------------------------------------------------- LOAD
dt <- fread(INPUT_CSV)
setnames(dt, old = intersect(c("map50", "map50_95"), names(dt)),
             new = c("mAP50", "mAP50_95")[c("map50", "map50_95") %in% names(dt)], skip_absent = TRUE)  # accept lowercase
req <- c("variant", "seed", "scene", "band", "severity", "realization", "method", PRIMARY_METRIC)
miss <- setdiff(req, names(dt))
if (length(miss)) stop("Input is missing columns: ", paste(miss, collapse = ", "))
dt[, variant := factor(as.character(variant), levels = VARIANT_ORDER)]
dt[, block := paste(scene, band, severity, realization, sep = "|")]

say("Rows: ", nrow(dt))
say("Rows per variant x seed:"); say_tbl(dcast(dt[, .N, by = .(variant, seed)], variant ~ seed, value.var = "N"))
say("Methods: ", paste(sort(unique(dt$method)), collapse = ", "))
if (!BASELINE_LABEL %in% dt$method) stop("BASELINE_LABEL '", BASELINE_LABEL, "' not found in method column")

clean_all <- dt[method == CLEAN_LABEL]
fact_all  <- dt[method != CLEAN_LABEL]
inf_dt    <- fact_all[seed == SEED_FOR_INFERENCE]
clean_inf <- clean_all[seed == SEED_FOR_INFERENCE]
if (nrow(inf_dt) == 0) stop("No rows for SEED_FOR_INFERENCE = ", SEED_FOR_INFERENCE)

# ----------------------------------------------------------------------------- HELPERS
bca_mean <- function(x, R = N_BOOT, conf = 0.95) {
  x <- x[is.finite(x)]
  if (length(x) < 3 || sd(x) == 0) return(c(lo = NA_real_, hi = NA_real_))
  b  <- boot(x, statistic = function(d, i) mean(d[i]), R = R)
  ci <- tryCatch(boot.ci(b, conf = conf, type = "bca")$bca[4:5], error = function(e) c(NA_real_, NA_real_))
  c(lo = ci[1], hi = ci[2])
}

# [REV3] Equivalence against the resolution bound (pooled table only).
# tost_equivalent: the 90% BCa interval of the mean delta lies inside (-RESOLUTION, +RESOLUTION),
# the confidence-interval-inclusion form of two one-sided tests at alpha = 0.05 (Schuirmann, 1987).
# tost_p: the paired-t form, the larger of the two one-sided p-values, for reference.
add_tost <- function(tbl, w) {
  res <- rbindlist(lapply(tbl$method, function(m) {
    x  <- w[[paste0("delta_", m)]]; x <- x[is.finite(x)]
    ci <- bca_mean(x, conf = 0.90)
    p  <- if (length(x) >= 2 && sd(x) > 0)
            max(t.test(x, mu = -RESOLUTION, alternative = "greater")$p.value,
                t.test(x, mu =  RESOLUTION, alternative = "less")$p.value) else NA_real_
    data.table(method = m, bca90_lo = ci[["lo"]], bca90_hi = ci[["hi"]], tost_p = p)
  }))
  tbl <- merge(tbl, res, by = "method", sort = FALSE)
  tbl[, tost_equivalent := is.finite(bca90_lo) & bca90_lo > -RESOLUTION & bca90_hi < RESOLUTION]
  tbl
}

paired_deltas <- function(d, metric) {
  w <- dcast(d, block + scene + band + severity + realization ~ method, value.var = metric)
  methods <- setdiff(unique(d$method), BASELINE_LABEL)
  for (m in methods) w[, (paste0("delta_", m)) := get(m) - get(BASELINE_LABEL)]
  list(wide = w, methods = methods)
}

delta_table <- function(w, methods) {
  rbindlist(lapply(methods, function(m) {
    x  <- w[[paste0("delta_", m)]]
    ci <- bca_mean(x)
    p  <- if (sum(is.finite(x)) >= 2 && sd(x, na.rm = TRUE) > 0)
            suppressWarnings(wilcox.test(x, mu = 0, exact = FALSE, correct = TRUE)$p.value) else NA_real_
    data.table(method = m, n_blocks = sum(is.finite(x)), mean_delta = mean(x, na.rm = TRUE),
               median_delta = median(x, na.rm = TRUE), bca_lo = ci[["lo"]], bca_hi = ci[["hi"]],
               wilcoxon_p = p)
  }))
}

friedman_block <- function(d, metric) {
  w <- dcast(d, block ~ method, value.var = metric)
  M <- as.matrix(w[, -1]); rownames(M) <- w$block
  M <- M[complete.cases(M), , drop = FALSE]
  fr <- friedman.test(M)
  ranks <- t(apply(M, 1, function(r) rank(-r)))      # rank 1 = best (highest mAP)
  mean_ranks <- sort(colMeans(ranks))
  nem <- tryCatch(frdAllPairsNemenyiTest(M), error = function(e) NULL)
  list(friedman = fr, mean_ranks = mean_ranks, nemenyi = nem, n_blocks = nrow(M))
}

# [REV2] per-condition headroom. Clean rows match on block if they carry severity and
# realization, otherwise on scene x band and are broadcast across severity x realization.
headroom_table <- function(d, clean_v, metric) {
  base <- d[method == BASELINE_LABEL, c(BLOCK_KEYS, metric), with = FALSE]
  setnames(base, metric, "D_degraded")
  cl <- clean_v[, c("scene", "band", "severity", "realization", metric), with = FALSE]
  setnames(cl, metric, "D_clean")
  # Rule: one clean row per scene x band (the praxis design: 12 per detector, severity == "clean")
  # -> broadcast across severity x realization. More than one per scene x band -> match per block.
  one_per_sb <- all(cl[, .N, by = .(scene, band)]$N == 1)
  j <- if (one_per_sb) merge(base, cl[, .(scene, band, D_clean)], by = c("scene", "band"), all.x = TRUE) else
         merge(base, cl, by = BLOCK_KEYS, all.x = TRUE)
  if (any(is.na(j$D_clean))) stop("headroom: ", sum(is.na(j$D_clean)),
      " baseline blocks have no clean-reference match; check CLEAN_LABEL and the clean rows' scene/band values")
  j[, headroom := D_clean - D_degraded]
  per_cond <- j[, .(n_blocks = .N, D_clean = mean(D_clean), D_degraded = mean(D_degraded),
                    headroom = mean(headroom), headroom_sd = sd(headroom)), by = .(band, severity)]
  setorder(per_cond, band, severity)
  list(per_block = j, per_cond = per_cond)
}

# ----------------------------------------------------------------------------- MAIN ANALYSIS PER VARIANT
results <- list()
for (v in VARIANT_ORDER) {
  d <- inf_dt[variant == v]
  if (nrow(d) == 0) { say("\n== Variant ", v, ": no rows, skipped =="); next }
  say("\n==================== Variant ", v, " (seed ", SEED_FOR_INFERENCE, ") ====================")

  pd <- paired_deltas(d, PRIMARY_METRIC)

  pooled <- delta_table(pd$wide, pd$methods)
  pooled[, p_bh := p.adjust(wilcoxon_p, method = "BH")]
  pooled[, beats_resolution := mean_delta > RESOLUTION & is.finite(bca_lo) & bca_lo > 0]
  pooled <- add_tost(pooled, pd$wide)                                # [REV3] equivalence at +/- RESOLUTION
  setorder(pooled, -mean_delta)
  say("\nPooled paired deltas vs baseline (", PRIMARY_METRIC, "), resolution = ", RESOLUTION, ":"); say_tbl(pooled)
  fwrite(pooled, file.path(OUT_DIR, sprintf("%s_pooled_deltas_%s.csv", v, PRIMARY_METRIC)))

  per_cond <- pd$wide[, delta_table(.SD, pd$methods), by = .(band, severity)]
  per_cond[, p_bh := p.adjust(wilcoxon_p, method = "BH")]           # BH across the whole per-condition family within the variant
  per_cond[, beats_resolution := mean_delta > RESOLUTION & is.finite(bca_lo) & bca_lo > 0]
  setorder(per_cond, band, severity, -mean_delta)
  fwrite(per_cond, file.path(OUT_DIR, sprintf("%s_per_condition_deltas_%s.csv", v, PRIMARY_METRIC)))

  if (SECONDARY_METRIC %in% names(d)) {
    pd2 <- paired_deltas(d, SECONDARY_METRIC)
    pooled2 <- delta_table(pd2$wide, pd2$methods)
    pooled2[, p_bh := p.adjust(wilcoxon_p, method = "BH")]
    fwrite(pooled2, file.path(OUT_DIR, sprintf("%s_pooled_deltas_%s.csv", v, SECONDARY_METRIC)))
  }

  fb <- friedman_block(d, PRIMARY_METRIC)
  say(sprintf("\nFriedman: chi2 = %.3f, df = %d, p = %.4g (n blocks = %d)",
              fb$friedman$statistic, fb$friedman$parameter, fb$friedman$p.value, fb$n_blocks))
  say("Mean ranks (1 = best):"); say_tbl(round(fb$mean_ranks, 3))
  if (!is.null(fb$nemenyi)) {
    say("Nemenyi post hoc p-values:"); say_tbl(fb$nemenyi)
    pm <- as.data.table(as.matrix(fb$nemenyi$p.value), keep.rownames = "method")       # [REV2] file output
    fwrite(pm, file.path(OUT_DIR, sprintf("%s_nemenyi.csv", v)))
  }
  fwrite(data.table(method = names(fb$mean_ranks), mean_rank = as.numeric(fb$mean_ranks)),
         file.path(OUT_DIR, sprintf("%s_mean_ranks.csv", v)))

  # [REV2] headroom
  hc <- NULL
  cv <- clean_inf[variant == v]
  if (nrow(cv)) {
    ht <- headroom_table(d, cv, PRIMARY_METRIC); hc <- ht$per_cond
    say("\nRecoverable headroom per condition (", PRIMARY_METRIC, "):"); say_tbl(hc)
    fwrite(hc, file.path(OUT_DIR, sprintf("%s_headroom_per_condition.csv", v)))
    fwrite(ht$per_block, file.path(OUT_DIR, sprintf("%s_headroom_per_block.csv", v)))
  } else say("\n[no clean-reference rows for this variant: headroom not computed]")

  b05 <- per_cond[band == B05_LABEL & severity == SEVERE_LABEL]
  if (!is.null(hc) && nrow(b05)) {
    h <- hc[band == B05_LABEL & severity == SEVERE_LABEL]
    b05[, headroom := if (nrow(h)) h$headroom[1] else NA_real_]
    b05[, delta_over_headroom := mean_delta / headroom]
  }
  say("\nB05 severe cell:"); if (nrow(b05)) say_tbl(b05) else say("cell not found")
  fwrite(b05, file.path(OUT_DIR, sprintf("%s_B05_severe.csv", v)))

  results[[v]] <- list(pooled = pooled, per_cond = per_cond, friedman = fb, b05 = b05, headroom = hc)
}

# ----------------------------------------------------------------------------- SECTION 3.8.7 CRITERIA
say("\n\n==================== Section 3.8.7 criteria ====================")
ref_order <- names(results[[REFERENCE_VARIANT]]$friedman$mean_ranks)
say("Reference ordering (", REFERENCE_VARIANT, "): ", paste(ref_order, collapse = " < "))

criteria <- rbindlist(lapply(names(results), function(v) {
  r  <- results[[v]]
  c1 <- r$pooled[beats_resolution == TRUE, method]
  pc <- r$per_cond[beats_resolution == TRUE]                                                   # [REV2]
  c1_cells <- if (nrow(pc)) sprintf("%s@%s/%s", pc$method, pc$band, pc$severity) else character(0)
  mr <- names(r$friedman$mean_ranks)
  b  <- r$b05
  c3 <- if (nrow(b)) sprintf("best method %s, delta %.4f [%.4f, %.4f], BH p %.3g, beats resolution %s%s",
                             b$method[1], b$mean_delta[1], b$bca_lo[1], b$bca_hi[1], b$p_bh[1],
                             as.character(b$beats_resolution[1]),
                             if ("headroom" %in% names(b)) sprintf(", headroom %.4f", b$headroom[1]) else "") else "cell not found"
  data.table(variant = v,
             crit1_pooled_any_beats = length(c1) > 0,
             crit1_pooled_methods   = paste(c1, collapse = ";"),
             crit1_any_cell_beats   = length(c1_cells) > 0,                                     # [REV2]
             crit1_cells            = paste(c1_cells, collapse = ";"),                           # [REV2]
             crit2_order_preserved  = identical(mr, ref_order),
             crit2_baseline_best_rank = mr[1] == BASELINE_LABEL,
             crit2_order            = paste(mr, collapse = " < "),
             crit3_B05_severe_any_beats = if (nrow(b)) any(b$beats_resolution) else FALSE,      # [REV2]
             crit3_B05_severe       = c3)
}))
say_tbl(criteria)
fwrite(criteria, file.path(OUT_DIR, "criteria_3_8_7.csv"))

# ----------------------------------------------------------------------------- SEED SPREAD (descriptive)
say("\n\n==================== Seed spread (descriptive, no test) ====================")
spread_base <- fact_all[method == BASELINE_LABEL, .(baseline_mAP = mean(get(PRIMARY_METRIC))), by = .(variant, seed)]
spread <- if (nrow(clean_all)) {
  merge(spread_base, clean_all[, .(clean_mAP = mean(get(PRIMARY_METRIC))), by = .(variant, seed)],
        by = c("variant", "seed"), all = TRUE)
} else spread_base
summ <- spread[, .(n_seeds = .N,
                   baseline_mean = mean(baseline_mAP), baseline_sd = sd(baseline_mAP),
                   clean_mean = if ("clean_mAP" %in% names(spread)) mean(clean_mAP) else NA_real_,
                   clean_sd   = if ("clean_mAP" %in% names(spread)) sd(clean_mAP) else NA_real_),
               by = variant][order(variant)]
say_tbl(summ)
fwrite(spread, file.path(OUT_DIR, "seed_values.csv"))
fwrite(summ,   file.path(OUT_DIR, "seed_spread.csv"))

if (all(c("m", "c") %in% as.character(summ$variant))) {
  m_sd <- summ[variant == "m", baseline_sd]
  diff <- summ[variant == "c", baseline_mean] - summ[variant == "m", baseline_mean]
  direction <- if (diff > 0) "c ABOVE m" else "c BELOW m"
  verdict <- if (is.na(m_sd)) "m SD unavailable (fewer than 2 seeds)" else
             if (abs(diff) <= m_sd) sprintf("INSIDE one m SD -> 'did not differ from' (%s)", direction) else
             sprintf("OUTSIDE one m SD -> %s, |diff|/m_SD = %.2f", direction, abs(diff) / m_sd)
  say(sprintf("\nm-to-c baseline difference = %.4f; m seed SD = %.4f; %s", diff, m_sd, verdict))
  writeLines(sprintf("m_to_c_diff=%.6f\nm_sd=%.6f\nverdict=%s", diff, m_sd, verdict),
             file.path(OUT_DIR, "m_vs_c_spread.txt"))
}
# NOTE for the text: seed spread for m and c mixes RTX 3090 and RTX A4500 runs (handoff 1.4); state it.

# ----------------------------------------------------------------------------- [REV2] REPRODUCTION CHECK
if (!is.na(PRIMARY_TBL) && file.exists(PRIMARY_TBL)) {
  say("\n\n==================== Reproduction check: m seed-0 vs Chapter 4 table ====================")
  prim <- fread(PRIMARY_TBL)
  setnames(prim, old = intersect(c("map50", "map50_95"), names(prim)),
                 new = c("mAP50", "mAP50_95")[c("map50", "map50_95") %in% names(prim)], skip_absent = TRUE)
  keys <- c(BLOCK_KEYS, "method")
  m0 <- dt[variant == "m" & seed == SEED_FOR_INFERENCE, c(keys, PRIMARY_METRIC), with = FALSE]
  pk <- prim[, c(keys, PRIMARY_METRIC), with = FALSE]; setnames(pk, PRIMARY_METRIC, "primary")
  j  <- merge(m0, pk, by = keys)
  j[, abs_diff := abs(get(PRIMARY_METRIC) - primary)]
  say(sprintf("matched rows: %d of %d; max |diff| = %.6f; rows with |diff| > 1e-6: %d",
              nrow(j), nrow(m0), max(j$abs_diff), sum(j$abs_diff > 1e-6)))
  fwrite(j, file.path(OUT_DIR, "reproduction_check_m_seed0.csv"))
}

say("\nOutputs written to ", normalizePath(OUT_DIR))
writeLines(capture.output(sessionInfo()), file.path(OUT_DIR, "sessionInfo.txt"))
close(log_con)
