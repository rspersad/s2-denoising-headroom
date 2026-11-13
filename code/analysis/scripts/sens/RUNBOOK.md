# Sensitivity arm (Section 3.8.7): runbook

Ordered. Each step names the host. Do not skip step 3.
Everything in this bundle was exercised end to end on a synthetic table with
the sweep schema; the R script was revised to match the Python outputs file
for file but could not be executed here (no R in the sandbox), so run it
through RStudio once on the synthetic table before pointing it at real data.

Files in this bundle

| File | Purpose |
|---|---|
| `assemble_sensitivity_table.py` | Concatenate the nine sweep CSVs, add `variant`/`seed`, validate, print the CONFIG labels |
| `stats_sensitivity.py` | Python implementation of the 3.8.7 analysis (Astraea) |
| `sensitivity_analysis.R` | R implementation, revision 2 (Nemesis, RStudio) |
| `compare_outputs.py` | Diff the two output directories with per-column tolerances |
| `make_synthetic.py` | Builds a fake table for exercising the scripts. NOT DATA. |
| `TEXT_ADDENDUM.md` | Sentences for Chapter 3 that the handoff surfaced |

---

## 1. Nemesis: confirm the seed queues finished, return the runs to Astraea

```bash
# Nemesis
grep -c '=== END' /data/praxis/seeds_nemesis.log        # expect 3
# Astraea
grep -c '=== END' /data/praxis/seeds_astraea.log        # expect 3
```

Both must show the expected count before anything else. Then, from Nemesis:

```bash
rsync -av --ignore-existing \
  /data/praxis/models/yolov9m-detection/tiled_p2_s1 \
  /data/praxis/models/yolov9m-detection/tiled_p2_s2 \
  /data/praxis/models/yolov9m-detection/tiled_p2_c_s1 \
  raj@10.20.10.30:/data/praxis/models/yolov9m-detection/
rsync -av /data/praxis/seeds_nemesis.log raj@10.20.10.30:/data/praxis/
```

Verify on Astraea that all nine run directories now hold `weights/best.pt`
and `results.csv`, then on Nemesis:

```bash
sudo zfs destroy tank/praxis/seeds
sudo sed -i 's/^REBOOT_ENABLED=0/REBOOT_ENABLED=1/' /etc/themis.conf
```

## 2. Astraea: extract the nine training summaries (for the Chapter 4 seed table)

```bash
cd /data/praxis/models/yolov9m-detection
python - <<'PY'
import csv
for r in ["tiled_p2","tiled_p2_s1","tiled_p2_s2","tiled_p2_c","tiled_p2_c_s1","tiled_p2_c_s2","tiled_p2_e","tiled_p2_e_s1","tiled_p2_e_s2"]:
    rows=[{k.strip():v for k,v in x.items()} for x in csv.DictReader(open(f"{r}/results.csv"))]
    b=max(rows,key=lambda x:float(x["metrics/mAP50-95(B)"]))
    print(f"{r:14s} epochs={len(rows):3d} best={b['epoch']:>3s} P={float(b['metrics/precision(B)']):.3f} R={float(b['metrics/recall(B)']):.3f} mAP50={float(b['metrics/mAP50(B)']):.3f} mAP50-95={float(b['metrics/mAP50-95(B)']):.4f}")
PY
```

These are validation-set training metrics. They go in the seed-spread
paragraph as descriptive context. They are not the sensitivity result.

## 3. Astraea: settle the two open facts about the primary table

```bash
ls -la /data/praxis/results /data/praxis/results_p2 /data/praxis/results_p2c
wc -l /data/praxis/results*/*.csv
head -3 /data/praxis/results*/*.csv
```

Then, for whichever file is the Chapter 4 table:

```bash
python - <<'PY'
import pandas as pd, sys
f = "/data/praxis/results_p2c/<the file>.csv"          # <-- set
d = pd.read_csv(f); print(len(d), "rows"); print(d.columns.tolist())
for c in ["scene","band","severity","method"]:
    if c in d: print(c, sorted(d[c].astype(str).unique()))
print(d.groupby([c for c in ["scene","band","severity"] if c in d]).size().unstack(fill_value=0))
PY
```

Record: the file path, the row count and how it reconciles with 648 + 12 +
144, and the exact labels in `method`, `band` and `severity`. The 516 in
paragraph 896 has to be explained by this step, not by the sensitivity
tables.

## 4. Astraea: read the sweep interface, then run the nine sweeps

```bash
cd ~/praxis/scripts
cat run_sensitivity_c.sh
grep -nE 'add_argument|argv|weights|\.pt|to_csv|columns|clean' evaluate_factorial.py | head -60
```

Mirror `run_sensitivity_c.sh` with only the weights path and the output
directory changed. One sweep per detector. Include the clean-reference
condition in every sweep or headroom cannot be computed.

```bash
mkdir -p /data/praxis/results_sens
for run in tiled_p2 tiled_p2_s1 tiled_p2_s2 tiled_p2_c tiled_p2_c_s1 tiled_p2_c_s2 tiled_p2_e tiled_p2_e_s1 tiled_p2_e_s2; do
  mkdir -p /data/praxis/results_sens/$run
  python evaluate_factorial.py \
      --weights /data/praxis/models/yolov9m-detection/$run/weights/best.pt \
      <same arguments as run_sensitivity_c.sh, severity restricted to nominal,severe, clean condition included> \
      --out /data/praxis/results_sens/$run/factorial_results.csv \
      2>&1 | tee /data/praxis/results_sens/$run/sweep.log
done
```

The argument names above are placeholders; take them from the grep. Split
across the 3090 and the A4500 if convenient, but write every table to
Astraea.

## 5. Astraea: assemble and validate

```bash
cd ~/praxis/scripts            # or wherever you put this bundle
python assemble_sensitivity_table.py \
    --root /data/praxis/results_sens \
    --out  /data/praxis/results_sens/all_variants.csv \
    --expect-conditions 36             # 54 if the sweeps ran the full grid
```

Must end with `VALIDATION: clean.` Copy the four CONFIG lines it prints into
the CONFIG block of **both** `stats_sensitivity.py` and
`sensitivity_analysis.R`. If it reports zero clean-reference rows, go back
to step 4.

## 6. Astraea: Python implementation, with the reproduction check

```bash
python stats_sensitivity.py \
    /data/praxis/results_sens/all_variants.csv \
    /data/praxis/results_sens/out_py \
    --primary /data/praxis/results_p2c/<the Chapter 4 table>.csv
```

Read `out_py/run_log.txt` first for the reproduction line. The m seed-0
rows must match the Chapter 4 table with `rows with |diff| > 1e-6: 0`.
If they do not, stop. Something differs between the primary sweep and this
one, and nothing downstream is interpretable until that is found.

## 7. Nemesis: R implementation

Copy `all_variants.csv` and the Chapter 4 table to Nemesis, open RStudio
Server, confirm `data.table`, `boot`, `PMCMRplus` are in the renv lockfile
(add `data.table` if the primary analysis did not use it), then:

```bash
Rscript sensitivity_analysis.R all_variants.csv out_r/ <chapter4_table.csv>
```

## 8. Either host: diff the two implementations

```bash
python compare_outputs.py out_r out_py
```

Must end with `R and Python outputs agree within tolerance.` BCa bounds are
allowed to differ by up to 0.0025 because the two bootstrap RNGs differ;
everything else is tight. A disagreement in `criteria_3_8_7.csv` is a
finding about the implementations, not about the detectors, and has to be
resolved before any number goes into Chapter 4.

## 9. Fill the Chapter 4 brackets

From `criteria_3_8_7.csv`, `seed_spread.csv`, `m_vs_c_spread.txt`, and the
three `*_B05_severe.csv` files, fill Part 7 of
`Detector_Ladder_Corrections.docx`. Then add the sentences in
`TEXT_ADDENDUM.md` to Chapter 3.

---

### Decisions encoded in the scripts that the text must state

- Inference is run on the seed-0 detectors only. Seeds 1 and 2 feed the
  descriptive spread and are never pooled as blocks.
- Criterion 1 is reported pooled across conditions and per cell. A
  capacity-dependent effect confined to one cell shows in the per-cell
  reading and not in the pooled one.
- Criterion 2 is reported two ways: strict identity of the full six-way
  ordering, and the weaker condition that the untreated baseline holds the
  best mean rank. Decide which 3.8.7 means before reading the numbers.
- BH correction is applied within variant, across the five methods for the
  pooled test and across all method-by-condition tests for the per-cell
  table. Confirm this matches the primary run's family definition in 3.8.3.
- No inferential test is run across variants.
