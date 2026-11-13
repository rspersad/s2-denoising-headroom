# Population-averaged logistic model: detected ~ band * log_area + severity + scene,
# cluster-robust SEs by vessel. Usage: Rscript scripts/vessel/pa_model.R <long.csv> <out_dir>
suppressPackageStartupMessages(library(sandwich))
a <- commandArgs(trailingOnly = TRUE); L <- read.csv(a[1]); outdir <- a[2]
L$band <- relevel(factor(L$band), ref = "B04"); L$severity <- factor(L$severity); L$scene <- factor(L$scene)
L$log_area_c <- L$log_area - mean(L$log_area)
m0 <- glm(detected ~ band + log_area_c + severity + scene, data = L, family = binomial)
m1 <- glm(detected ~ band * log_area_c + severity + scene, data = L, family = binomial)
V1 <- vcovCL(m1, cluster = L$vessel_id); V0 <- vcovCL(m0, cluster = L$vessel_id)
wald <- function(m, V, terms) { b <- coef(m)[terms]; W <- as.numeric(t(b) %*% solve(V[terms, terms]) %*% b); c(W = W, df = length(terms), p = pchisq(W, length(terms), lower.tail = FALSE)) }
ors <- function(m, V) { b <- coef(m); se <- sqrt(diag(V)); round(cbind(OR = exp(b), lo = exp(b - 1.96 * se), hi = exp(b + 1.96 * se), p = 2 * pnorm(-abs(b / se))), 3) }
sink(file.path(outdir, "pa_model.txt"))
cat("Rows:", nrow(L), " vessels:", length(unique(L$vessel_id)), "\n\n")
cat("=== Cluster-robust Wald test (by vessel) of the band x log_area interaction ===\n"); print(wald(m1, V1, c("bandB05:log_area_c", "bandB11:log_area_c")))
cat("\n=== m1 odds ratios, cluster-robust 95% intervals ===\n"); print(ors(m1, V1))
cat("\n=== m0 (no interaction) odds ratios, cluster-robust 95% intervals ===\n"); print(ors(m0, V0))
cat("\n=== Predicted detection probability by band at tercile midpoints (m1, nominal severity, averaged over scenes) ===\n")
q <- quantile(exp(L$log_area), c(0.17, 0.5, 0.83)); sev <- if ("nominal" %in% levels(L$severity)) "nominal" else levels(L$severity)[1]
nd <- expand.grid(band = levels(L$band), area = as.numeric(q), severity = sev, scene = levels(L$scene))
nd$log_area_c <- log(nd$area) - mean(L$log_area); nd$p <- predict(m1, newdata = nd, type = "response")
agg <- aggregate(p ~ band + area, data = nd, FUN = mean)
print(round(reshape(agg, idvar = "area", timevar = "band", direction = "wide"), 3))
sink(); cat(readLines(file.path(outdir, "pa_model.txt")), sep = "\n")
