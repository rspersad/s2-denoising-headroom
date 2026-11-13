#!/usr/bin/env bash
# Detector sensitivity arm (3.8.7): frozen-weight factorial sweep for all nine
# detectors (m/c/e x seeds 0-2), full 54-condition grid plus clean reference.
# Recipe identical to run_sensitivity_c.sh except weights and output per run.
set -u
cd "$(dirname "$0")/.."
MODELS=/data/praxis/models/yolov9m-detection
OUT_ROOT=/data/praxis/results_sens
mkdir -p "$OUT_ROOT"
echo "=== QUEUE START $(date)"

COMMON=(--manifest /data/praxis/processed/s2ships_manifest.csv
        --coco     /data/praxis/processed/s2ships_exploded_coco.json
        --bounds   /data/praxis/detection_tiled/stretch_bounds.json
        --degraded /data/praxis/synthetic-degradations/sentinel2
        --denoised /data/praxis/denoised)

for RUN in tiled_p2 tiled_p2_s1 tiled_p2_s2 \
           tiled_p2_c tiled_p2_c_s1 tiled_p2_c_s2 \
           tiled_p2_e tiled_p2_e_s1 tiled_p2_e_s2; do
  W="$MODELS/$RUN/weights/best.pt"
  OUT="$OUT_ROOT/$RUN"
  if [ ! -f "$W" ]; then echo "=== MISSING $W $(date)"; continue; fi
  mkdir -p "$OUT"
  echo "=== START $RUN $(date)"
  sha1sum "$W" | tee "$OUT/detector_sha1.txt"
  python3 scripts/evaluate_factorial.py "${COMMON[@]}" --weights "$W" --out "$OUT" \
    && echo "=== END $RUN $(date) rows=$(($(wc -l < "$OUT/factorial_results.csv") - 1))" \
    || echo "=== FAILED $RUN $(date)"
done
echo "=== QUEUE END $(date)"
