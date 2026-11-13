#!/usr/bin/env bash
# Assemble the public repository and the Zenodo archive from the nemesis copies.
# Non-destructive: reads from /tank/praxis and ~/s2_analysis, writes only under ~/s2-denoising-headroom
# and ~/zenodo-staging. Does NOT create the GitHub repository or push anything.
set -euo pipefail

B=/tank/praxis/astraea-backup
P=/tank/praxis/repo
A=~/s2_analysis
R=~/s2-denoising-headroom
Z=~/zenodo-staging
SCAFFOLD=~/s2-denoising-headroom_scaffold.tar.gz

[ -f "$SCAFFOLD" ] || { echo "missing $SCAFFOLD — scp it from Circe first"; exit 1; }
mkdir -p "$R" "$Z"
cd "$R"
tar xzf "$SCAFFOLD"
[ -d .git ] || git init -b main -q

echo "== code"
mkdir -p code/pipeline code/analysis
rsync -a --exclude '.git' --exclude '*.pt' --exclude '*.pth' --exclude 'renv/library' --exclude 'renv/staging' \
      --exclude 'auth_test.py' --exclude '__pycache__' --exclude '.Rhistory' --exclude '.env' \
      "$P/scripts/" code/pipeline/
rsync -a --exclude '.git' --exclude '__pycache__' --exclude '.Rhistory' --exclude '.env' "$A/" code/analysis/
# record the exact commits the code was taken from
{ echo "pipeline (rspersad/praxis): $(git -C "$P" rev-parse HEAD) on $(git -C "$P" rev-parse --abbrev-ref HEAD)";
  echo "analysis (s2_analysis):     $(git -C "$A" rev-parse HEAD) on $(git -C "$A" rev-parse --abbrev-ref HEAD)"; } > code/SOURCE_COMMITS.txt
cat code/SOURCE_COMMITS.txt

echo "== results tables"
mkdir -p results/primary results/sensitivity results/vessel results/superseded
cp -r "$B/results_p2c/." results/primary/
cp -r "$B/results_sens/." results/sensitivity/
cp -r "$B/results_vessel/." results/vessel/
cp -r "$B/results_p2" results/superseded/results_p2
cp -r "$B/results"    results/superseded/results_pre_correction
echo "row counts (header included):"
for f in "$B"/results*/factorial_results*.csv; do printf "  %6d  %s\n" "$(wc -l < "$f")" "$f"; done

echo "== configs, splits, manifests, logs"
mkdir -p configs docs/logs
cp "$B/synthetic-degradations/sentinel2/degradation_manifest.csv" configs/
cp -r "$B/splits" configs/splits
cp -r "$B/manifests" configs/manifests
cp "$B"/*.log docs/logs/
cp "$B/results_p2c/detector_sha1.txt" configs/ 2>/dev/null || true

echo "== Zenodo staging (large files)"
# tiles and degraded/denoised variants; label files packed separately so they can be withheld pending the S2-SHIPS licence check
tar -I 'zstd -T0 -10' -cf "$Z/tiles_clean.tar.zst"        -C "$B" processed detection_tiled --exclude='*/labels/*'
tar -I 'zstd -T0 -10' -cf "$Z/labels_yolo.tar.zst"        -C "$B" --wildcards 'detection_tiled/*/labels' 2>/dev/null || \
  find "$B/detection_tiled" -type d -name labels -printf '%P\n' | tar -I 'zstd -T0 -10' -cf "$Z/labels_yolo.tar.zst" -C "$B/detection_tiled" -T -
tar -I 'zstd -T0 -10' -cf "$Z/tiles_degraded.tar.zst"     -C "$B" synthetic-degradations
tar -I 'zstd -T0 -10' -cf "$Z/tiles_denoised.tar.zst"     -C "$B" denoised denoised_matched
# the nine detectors of record, renamed by scale and seed
mkdir -p "$Z/detectors"
for s in m c e; do
  case $s in m) d=tiled_p2;; c) d=tiled_p2_c;; e) d=tiled_p2_e;; esac
  cp "$B/models/yolov9m-detection/$d/weights/best.pt"    "$Z/detectors/yolov9${s}_seed0.pt"
  cp "$B/models/yolov9m-detection/${d}_s1/weights/best.pt" "$Z/detectors/yolov9${s}_seed1.pt"
  cp "$B/models/yolov9m-detection/${d}_s2/weights/best.pt" "$Z/detectors/yolov9${s}_seed2.pt"
done
tar -I 'zstd -T0 -10' -cf "$Z/detectors_yolov9.tar.zst" -C "$Z" detectors && rm -rf "$Z/detectors"
cp LICENSE "$Z/LICENSE-detectors-AGPL-3.0.txt"; cp LICENSE-DATA "$Z/LICENSE-data-CC-BY-4.0.txt"; cp NOTICE.md README.md "$Z/"
ls -lh "$Z"

echo "== manifest"
mkdir -p data/archive && ln -sfn "$Z" data/archive/zenodo
python3 tools/make_manifest.py code results configs docs "$Z"
sed -i "s#$Z#data/archive#g" MANIFEST.sha256

echo "== privacy scan (must be empty before you publish)"
grep -rn -i -E "token|passw|secret|client_id|api[_-]?key|@gmail|/home/raj" code results configs docs --include='*.py' --include='*.R' --include='*.yaml' --include='*.yml' --include='*.json' --include='*.cfg' --include='*.txt' --include='*.md' --include='*.log' | grep -v -E "CITATION|NOTICE|README" | head -40 || true
echo "== done. Review the scan above, then: cd $R && git status | head"
