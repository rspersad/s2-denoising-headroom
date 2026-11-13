#!/usr/bin/env bash
# H3 severity-matched sweep (DnCNN, Restormer fine-tuned at applied severity) for all nine detectors.
set -u
cd "$(dirname "$0")/.."
MODELS=/data/praxis/models/yolov9m-detection
OUT_ROOT=/data/praxis/results_sens
echo "=== MATCHED QUEUE START $(date)"
COMMON=(--manifest /data/praxis/processed/s2ships_manifest.csv
        --coco     /data/praxis/processed/s2ships_exploded_coco.json
        --bounds   /data/praxis/detection_tiled/stretch_bounds.json
        --degraded /data/praxis/synthetic-degradations/sentinel2
        --denoised /data/praxis/denoised_matched)
for RUN in tiled_p2 tiled_p2_s1 tiled_p2_s2 tiled_p2_c tiled_p2_c_s1 tiled_p2_c_s2 tiled_p2_e tiled_p2_e_s1 tiled_p2_e_s2; do
  W="$MODELS/$RUN/weights/best.pt"; OUT="$OUT_ROOT/$RUN/matched"; mkdir -p "$OUT"
  echo "=== START $RUN $(date)"
  python3 scripts/evaluate_factorial.py "${COMMON[@]}" --weights "$W" --out "$OUT" \
      --methods dncnn restormer --severities moderate severe \
    && echo "=== END $RUN $(date) rows=$(($(wc -l < "$OUT/factorial_results.csv") - 1))" \
    || echo "=== FAILED $RUN $(date)"
done
echo "=== MATCHED QUEUE END $(date)"
