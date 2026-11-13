# Vessel-level test of H2 — runbook

Post hoc analysis, declared as such in the deviations register. Unit of analysis is the annotated vessel (paired across the three bands), not the scene. Uses the primary detector (`tiled_p2`, seed 0) and only the clean and untreated conditions; no denoising runs are involved.

Files: `patch_evaluate_factorial.py`, `score_vessels.py`, `analyze_vessels.py`. Put them in `scripts/vessel/` in the repository.

## 1. Astraea: patch the evaluator (idempotent, keeps a .bak)

```bash
cd ~/praxis
mkdir -p scripts/vessel && cp /path/to/unpacked/*.py scripts/vessel/
python scripts/vessel/patch_evaluate_factorial.py scripts/evaluate_factorial.py
git diff --stat scripts/evaluate_factorial.py
```

The diff should touch three places: a new `--save-pred` argument, `save_json=bool(args.save_pred)` on the `model.val` call, and a block after `map5095 = ...` that copies `predictions.json`, the tile labels and a tile list out of the temp directory. Nothing else changes, so the factorial numbers are unaffected when `--save-pred` is not given.

## 2. Astraea: re-run inference on the clean and untreated conditions only

Take every argument except `--weights`, `--methods`, `--out` and `--save-pred` verbatim from `scripts/run_sensitivity_c.sh` (manifest, coco, bounds, degraded, denoised, conf, iou, device).

```bash
cd ~/praxis/scripts
mkdir -p /data/praxis/results_vessel
python evaluate_factorial.py \
    --weights /data/praxis/models/yolov9m-detection/tiled_p2/weights/best.pt \
    --methods clean none \
    <manifest / coco / bounds / degraded / denoised / conf / iou / device, as in run_sensitivity_c.sh> \
    --out /data/praxis/results_vessel/factorial_clean_none.csv \
    --save-pred /data/praxis/results_vessel/pred \
    2>&1 | tee /data/praxis/results_vessel/sweep.log
ls /data/praxis/results_vessel/pred
ls /data/praxis/results_vessel/pred/none/rome/B04
```

Expect `clean` and `none` under `pred/`, and under `none/<scene>/<band>/` nine directories `nominal_r0 … severe_r2`, each holding `predictions.json`, `labels/` and `tiles.txt`. If `clean` does not appear, the evaluator handles the clean reference outside the `--methods` list; tell me how the clean rows are produced and I will adjust.

Check that the metrics did not move: the `none` and `clean` rows of `factorial_clean_none.csv` must equal the corresponding rows of `results/primary/factorial_results.csv` (`save_json` does not change inference).

```bash
python - <<'PY'
import pandas as pd
a = pd.read_csv("/data/praxis/results_vessel/factorial_clean_none.csv")
b = pd.read_csv("~/praxis/results/primary/factorial_results.csv"); b = b[b.method.isin(["clean","none"])]
k = ["method","band","severity","realization","scene"]
m = a.merge(b, on=k, suffixes=("_new","_primary"))
print(len(a), len(b), len(m), "max |map50 diff| =", (m.map50_new - m.map50_primary).abs().max())
PY
```

## 3. Astraea: score vessels and run the tests

```bash
cd ~/praxis
python -I scripts/vessel/score_vessels.py --pred /data/praxis/results_vessel/pred --out results/vessel/vessels.csv
python -I scripts/vessel/analyze_vessels.py --vessels results/vessel/vessels.csv --out results/vessel/out --thr 0.25
```

`score_vessels.py` prints one line per condition with the vessel count (expect 388 ± a few across the four scenes once duplicates across overlapping tiles are merged; vessels cut by every tile edge are absent, as they are from the mAP evaluation). `analyze_vessels.py` prints the band tests per condition, per scene and per size tercile, and writes `results/vessel/out/vessel_band_tests.csv` and `vessel_long_untreated.csv`.

## 4. Nemesis: the mixed model in R

```r
# Nemesis, RStudio, in the renv project
library(lme4)
L <- read.csv("results/vessel/out/vessel_long_untreated.csv")
L$band <- relevel(factor(L$band), ref = "B04")
m0 <- glmer(detected ~ band + log_area + severity + (1 | scene) + (1 | vessel_id), data = L, family = binomial)
m1 <- glmer(detected ~ band * log_area + severity + (1 | scene) + (1 | vessel_id), data = L, family = binomial)
anova(m0, m1)      # does the band effect depend on vessel size?
summary(m1)
```

Keep `scene` as a random intercept even with four levels; it is there to absorb the scene effect, not to estimate it. Report the fixed-effect odds ratios for band with Wald intervals, the likelihood-ratio test for the interaction, and the per-scene detection rates from step 3 beside them.

## 5. Commit

```bash
cd ~/praxis
git add scripts/vessel scripts/evaluate_factorial.py results/vessel
git commit -m "Vessel-level H2 analysis: --save-pred on evaluate_factorial, per-vessel scoring, band tests and mixed-model inputs"
git push origin correction/detector-tiled-p2
```

## What goes in the praxis

- 3.8.6, deviations register: "H2 was additionally tested post hoc at the level of the annotated vessel, paired across bands, by Cochran's Q and McNemar tests on detection at a fixed confidence of 0.25, by Friedman and Page tests on matched confidence, and by a mixed-effects logistic regression of detection on band, vessel size and severity with scene and vessel random intercepts. The Friedman result of the pre-specified analysis is reported unchanged."
- 4.2.1: one paragraph with the vessel-level result, whatever it is, and the band × size interaction if it is there.
- 1.8: the three-locations-plus-one sentence about Rotterdam.
