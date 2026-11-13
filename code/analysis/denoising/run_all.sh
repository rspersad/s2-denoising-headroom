#!/usr/bin/env bash
# Launch all learned-denoiser training jobs on Astraea (single RTX 3090).
#
# Runs sequentially by default (one GPU). Each job logs to $LOG_DIR and writes
# a DONE marker on completion so re-runs skip finished jobs. Designed to live in
# a tmux session you can detach from over mosh:
#
#   tmux new -s denoisers
#   bash denoising/run_all.sh
#   # Ctrl-b d to detach; `tmux attach -t denoisers` to return
#
# Override what runs:
#   METHODS="dncnn noise2void" BANDS="B04" bash denoising/run_all.sh
#
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1   # repo root (parent of denoising/)

BANDS="${BANDS:-B04 B05 B11}"
SEVERITIES="${SEVERITIES:-nominal moderate severe}"
METHODS="${METHODS:-dncnn restormer noise2void blind2unblind}"
RESTORMER_ARCH="${RESTORMER_ARCH:-external/Restormer/basicsr/models/archs/restormer_arch.py}"
LOG_DIR="${LOG_DIR:-logs/denoisers}"
PY="${PY:-python}"

mkdir -p "$LOG_DIR"

run() {  # method band severity extra-args...
  local method="$1" band="$2" sev="$3"; shift 3
  local name="${method}_${band}_${sev}"
  local done_marker="${PRAXIS_CKPT_ROOT:-/data/praxis/models/denoisers}/${method}/${band}_${sev}/DONE"
  if [[ -f "$done_marker" ]]; then
    echo "[skip] $name (DONE exists)"; return 0
  fi
  echo "[run ] $name -> $LOG_DIR/$name.log"
  $PY -m "denoising.train_${method}" --band "$band" --severity "$sev" "$@" \
      >"$LOG_DIR/$name.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then echo "[FAIL] $name (rc=$rc) -- see log"; fi
  return $rc
}

echo "Bands=$BANDS  Severities=$SEVERITIES  Methods=$METHODS"
START=$(date +%s)

for method in $METHODS; do
  case "$method" in
    dncnn)
      # supervised: nominal severity only, no --severity flag accepted
      for band in $BANDS; do
        name="dncnn_${band}_nominal"
        done_marker="${PRAXIS_CKPT_ROOT:-/data/praxis/models/denoisers}/dncnn/${band}_nominal/DONE"
        [[ -f "$done_marker" ]] && { echo "[skip] $name"; continue; }
        echo "[run ] $name"
        $PY -m denoising.train_dncnn --band "$band" >"$LOG_DIR/$name.log" 2>&1 \
          || echo "[FAIL] $name -- see log"
      done
      ;;
    restormer)
      for band in $BANDS; do
        name="restormer_${band}_nominal"
        done_marker="${PRAXIS_CKPT_ROOT:-/data/praxis/models/denoisers}/restormer/${band}_nominal/DONE"
        [[ -f "$done_marker" ]] && { echo "[skip] $name"; continue; }
        echo "[run ] $name"
        $PY -m denoising.train_restormer --band "$band" --arch "$RESTORMER_ARCH" \
          >"$LOG_DIR/$name.log" 2>&1 || echo "[FAIL] $name -- see log"
      done
      ;;
    noise2void|blind2unblind)
      for band in $BANDS; do
        for sev in $SEVERITIES; do
          run "$method" "$band" "$sev"
        done
      done
      ;;
    *)
      echo "[warn] unknown method '$method'";;
  esac
done

echo "All requested jobs processed in $(( $(date +%s) - START ))s."
