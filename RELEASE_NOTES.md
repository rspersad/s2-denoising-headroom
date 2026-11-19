# v1.0.1 — Cochran's Q correction

Corrects `code/analysis/scripts/vessel/analyze_vessels.py`, which computed Cochran's Q with the concordant vessels removed from the row sums of the denominator but not from the numerator or the total, understating Q wherever concordant vessels were numerous (praxis repository commit 4a872cc). Every term is now computed on the full matrix.

* `code/analysis/results/vessel/out/vessel_band_tests.csv` regenerated; detection fractions, McNemar, Friedman and Page columns unchanged. Corresponds to manuscript Table 4.2 from v1.15 onward.
* `code/analysis/results/native_tiling/{run640,run1280,train1280}/out/vessel_band_tests.csv` regenerated with the same script.
* No data, labels or detector weights changed; the archive record doi:10.5281/zenodo.24209175 is unaffected.

# v1.0.0 — data and code of record

Corresponds to the Praxis as approved by the committee on 12 November 2026 (manuscript v1.12).

* `results/primary_table.csv` — 660 rows: 648 factorial evaluations and 12 clean-reference evaluations.
* `results/severity_matched.csv` — 144 evaluations of the severity-matched arm.
* `results/calibration.csv` — Table 3.1 values and the reference-radiance note.
* `results/ceiling_contrast.csv` — the ceiling-contrast table referenced in Chapter 4 (repository commit bfd6d7b).
* Large files (tiles, labels, detectors) — archive record doi:10.5281/zenodo.24209175.

Every file is listed with its SHA-256 in `MANIFEST.sha256`.
