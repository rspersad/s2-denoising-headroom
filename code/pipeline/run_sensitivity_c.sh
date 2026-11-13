#!/usr/bin/env bash
# Detector sensitivity arm (3.9.7): yolov9c + P2 head, recipe identical to tiled_p2.
set -u
cd "$(dirname "$0")/.."
OUT=/data/praxis/results_p2c
MODELS=/data/praxis/models/yolov9m-detection
mkdir -p "$OUT/matched"
echo "=== START $(date)"

if python3 scripts/train_p2.py --model yolov9c-p2.yaml --pretrained yolov9c.pt --name tiled_p2_c --batch 16; then
  RUN=tiled_p2_c
elif python3 scripts/train_p2.py --model yolov9c-p2.yaml --pretrained yolov9c.pt --name tiled_p2_c_b8 --batch 8; then
  RUN=tiled_p2_c_b8
else
  echo "=== TRAINING FAILED at batch 16 and 8 $(date)"; exit 1
fi
W="$MODELS/$RUN/weights/best.pt"
echo "=== TRAINED $RUN  $(date)"; sha1sum "$W" | tee "$OUT/detector_sha1.txt"

COMMON=(--manifest /data/praxis/processed/s2ships_manifest.csv
        --coco /data/praxis/processed/s2ships_exploded_coco.json
        --bounds /data/praxis/detection_tiled/stretch_bounds.json
        --weights "$W"
        --degraded /data/praxis/synthetic-degradations/sentinel2)

python3 scripts/evaluate_factorial.py "${COMMON[@]}" --denoised /data/praxis/denoised --out "$OUT" \
  && echo "=== PRIMARY DONE $(date)"
python3 scripts/evaluate_factorial.py "${COMMON[@]}" --denoised /data/praxis/denoised_matched --out "$OUT/matched" \
  --methods dncnn restormer --severities moderate severe \
  && echo "=== MATCHED DONE $(date)"

wc -l "$OUT/factorial_results.csv" "$OUT/matched/factorial_results.csv"
echo "=== END $(date)"
