# Learned-denoiser training (`denoising/`)

Trains the four learned denoisers for the praxis factorial, all built on the
Appendix C Poisson-Gaussian noise model and the §3.5 configurations.

## Job matrix (24 models)

| Method | Paradigm | Trained per | Severity | Count |
|---|---|---|---|---|
| DnCNN | supervised | band | nominal only | 3 |
| Restormer | supervised (fine-tune) | band | nominal only | 3 |
| Noise2Void | self-supervised | band × severity | all 3 | 9 |
| Blind2Unblind | self-supervised | band × severity | all 3 | 9 |

Supervised models train on **nominal**-severity clean/noisy pairs only — this is
the deliberate "train-clean, test-shifted" setup H3 examines. Self-supervised
models retrain at each severity from that severity's noisy imagery.

## Run order

```bash
# 0. deps (into the 'pacific' env)
pip install -r denoising/requirements.txt

# 1. AUDIT FIRST — verifies paths, partition, band indices, value ranges, noise stats
python -m denoising.audit_data

# 2. (Restormer only) vendor the official arch + grayscale checkpoint
git clone https://github.com/swz30/Restormer external/Restormer
#   download Gaussian_Gray_Denoising blind weights -> config.RESTORMER_PRETRAINED

# 3. launch everything in tmux (single GPU, sequential, resumable)
tmux new -s denoisers
bash denoising/run_all.sh        # METHODS=/BANDS=/SEVERITIES= to subset
```

`run_all.sh` writes a `DONE` marker per finished job and skips it on re-run, so
a dropped mosh session just means `tmux attach` and (if needed) re-run.

## What `audit_data.py` confirms (and you should eyeball)

Everything layout-dependent is centralized in `config.py`. The audit reports the
**actual** values so you can reconcile them:

- **Manifest / partition** — reads `s2ships_manifest.csv`. Denoiser training uses
  the 12 non-eval scenes; the 4 `EVAL_SCENES` (panama, rome, brest1, rotterdam1)
  are held out of all denoiser training to prevent factorial leakage. The audit
  prints the resolved split so you can confirm the right scenes are excluded.
- **Band rasters** — paths come straight from the manifest's `path_B04/B05/B11`
  columns (single-band GeoTIFFs), so there is no band-index guessing.
- **`REFLECTANCE_SCALE`** — noise is applied in DN (Decision D12); input is
  divided by 10000 (the manifest's quantification value) for the network. The
  audit shows the actual value range so you can confirm.

### Partition (three-way design, Chapter 3 §3.6.2 as revised)

| Role | Scenes | Used by |
|---|---|---|
| Train | suez1-6, toulon, rotterdam3, southampton, portsmouth (10) | YOLO train + denoisers |
| Val | marseille, rotterdam2 (2) | YOLO checkpoint selection + denoisers |
| Eval | panama, rome, brest1, rotterdam1 (4) | factorial only — held out everywhere |

Denoisers train on the 12 non-eval scenes (train+val). Factorial = 54 conditions
× 3 realizations × 4 eval scenes = 648 evaluations.

## Fidelity notes (worth a look before you commit results to the thesis)

These three points are where the implementation makes a choice you may want to
align with the Chapter 3 wording:

1. **Noise2Void** — §3.5.4 names the `n2v` library (Weigert et al.). `n2v` is
   TensorFlow/Keras (CSBDeep), which doesn't sit cleanly alongside the torch
   DnCNN/Restormer/B2U stack on one box. `train_noise2void.py` is a faithful
   torch reimplementation of the N2V blind-spot scheme (random masking +
   neighbour replacement + masked MSE). Either keep it and adjust the chapter to
   "a reimplementation of the N2V scheme," or run the literal library in a
   separate TF env. Your call — flag for the advisor either way.

2. **Blind2Unblind** — §3.5.5 names the authors' reference implementation.
   `train_blind2unblind.py` reimplements the global-mask + re-visible scheme in
   torch. **This is the subtlest loss in the pipeline.** Recommend validating it
   numerically against `zejinwang/Blind2Unblind`, or swapping that repo in at the
   `loss_step` boundary. The `CELL`, `LAMBDA_MAX`, `RAMP_EPOCHS` constants are the
   knobs to check.

3. **Restormer** — §3.5.3 initializes from the official released checkpoint, so
   we use the authors' architecture (not a reimplementation) to keep keys
   compatible. Defaults to the **grayscale** blind denoising weights since we
   operate on single bands.

## Files

```
denoising/
  config.py              # paths, band map, severity factors, hyperparameters
  noise_model.py         # Appendix C Poisson-Gaussian + deterministic seeding
  data.py                # partition resolution, band reads, patch datasets
  masking.py             # N2V / B2U blind-spot masking
  models/__init__.py     # DnCNN + U-Net (Restormer uses official arch)
  train_common.py        # shared loop: cosine LR, AMP, checkpoint, logging
  train_dncnn.py         # supervised, per-band, nominal
  train_restormer.py     # supervised fine-tune, per-band, nominal
  train_noise2void.py    # self-supervised, per-band, per-severity
  train_blind2unblind.py # self-supervised, per-band, per-severity
  audit_data.py          # RUN FIRST
  run_all.sh             # tmux launcher (24 jobs, resumable)
  requirements.txt
```

Checkpoints land in `$PRAXIS_CKPT_ROOT/<method>/<band>_<severity>/best.pth`
(default `/data/praxis/models/denoisers/...`), ready for the inference pass that
feeds the synthetic-degradation factorial.
