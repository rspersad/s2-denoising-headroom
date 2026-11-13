# Praxis Progress Log — Compute, Reproducibility Pipeline, and Detection Development

**Praxis:** Perceptual Quality Versus Detection Performance: An Evaluation of AI Denoising
Methods for Sentinel-2 Vessel Detection
**Thesis:** Perceptual image quality and downstream detection performance are decoupled
constructs; preprocessing and denoising choices should be made on task-aware rather than
perceptual grounds.
**Benchmark:** S2-SHIPS · **Detector:** YOLOv9m · **Metric:** mAP

---

## 1. Compute Environment

A three-node k3s cluster (v1.35.5+k3s1, frozen for the duration of the study) plus direct
GPU execution:

| Node | Role | Hardware | Use |
|---|---|---|---|
| Phoebe | control-plane | i7-7700, 16 GB, no GPU | kubectl entry point |
| Nemesis | cpu-worker | Xeon Platinum | CPU cluster Jobs (e.g. BM3D) |
| Astraea | gpu-worker | RTX 3090 (driver 550.163.01, CUDA 12.4), Debian 13 | GPU work via conda env `pacific`; data host |

- **Storage:** Astraea exports `/data` over NFS to Nemesis and Phoebe, so CPU Jobs on
  Nemesis read the same data the GPU work produces on Astraea.
- **Containers:** Docker on Astraea only. Images are built on Astraea, exported
  (`docker save | gzip`), copied to Nemesis, and imported into k3s containerd
  (`k3s ctr -n k8s.io images import`). Jobs use `imagePullPolicy: Never`.
- **GPU work runs off-cluster** directly in the `pacific` conda env (torch 2.4.0+cu121,
  ultralytics 8.4.51, rasterio 1.4.4, cv2 4.13.0); the GPU is deliberately kept out of the
  k8s scheduling pool.
- Full detail recorded in **Appendix D (Computational Environment)**.

---

## 2. Reproducibility Pipeline (complete and verified)

### 2.1 Preprocessing (validated)
S2-SHIPS annotations exploded to 951 single-class vessel boxes; 16-scene manifest with
per-band paths, water masks, and dimensions (938 × 1783); Poisson–Gaussian noise
calibration per band (B04 α=8.42/σr=5.0, B05 α=7.97/σr=5.0, B11 α=19.12/σr=2.0) at three
severities. Verified against the documented spec (e.g., Toulon = 261 boxes).

### 2.2 Synthetic Degradation (complete, reproducible)
Model y = (1/α)·Poisson(α·I) + N(0, σr²), SHA-256-seeded per (scene, band, severity,
realization). **432 images** (16 × 3 × 3 × 3). Run as cluster Job `degrade-s2ships`
(image `praxis-cpu:0.1`). Proven **pixel-identical** between the conda env and the
container (pixel digest `d196a528…`). Provenance: `degradation_provenance.md`.

### 2.3 BM3D Denoising (complete, validated)
Poisson–Gaussian denoising via the generalized Anscombe transform:
forward GAT → BM3D (σ=1) → closed-form unbiased inverse GAT. **108 images** (4 eval scenes
× 3 × 3 × 3; BM3D needs no training, so only the evaluation partition is denoised). Run as
cluster Job `bm3d-denoise` on Nemesis over NFS (image `praxis-cpu:0.2`, bm3d 4.0.3).

**Correctness fix recorded:** the reference GAT forward/inverse pair is self-consistent
only at unit gain. Our model has gain = 1/α ≠ 1, so the data are standardized by z = α·y
(Poisson-count scale, σz = α·σr), transformed there, and divided by α on return.
Round-trip recovers a clean signal to ~2×10⁻⁵ mean relative error; the naive gain≠1
application carried a ~σ²/gain bias and was discarded. Validation on Panama B11/severe:
RMSE-to-clean 37.11 → 30.13 (~19%). Provenance: `bm3d_provenance.md`.

### Evaluation partition (corrects Ch.3 §3.6.2 wording — Suez is TRAINING)
- **Train (10):** suez1–6, toulon, rotterdam3, southampton, portsmouth
- **Val (2):** marseille, rotterdam2
- **Eval (4, held out):** panama, rome, brest1, rotterdam1
- 648 evaluations = 54 conditions × 4 eval scenes × 3 realizations.

---

## 3. Detection Development

### 3.1 Architecture/data progression
The detector requires resolving vessels of median ~8×5 px. Development proceeded by staged,
cheap-to-validate changes:

| Run | Change | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| clean_baseline-2 | stock YOLOv9m, whole scene letterboxed 1783→640 | 0.326 | 0.100 | 0.069 | 0.021 |
| tiled_baseline | 640×640 native-resolution tiles (stride 512) | 0.462 | 0.205 | 0.184 | 0.083 |
| tiled_p2 | + P2/4 small-object detection head | 0.522 | 0.167 | 0.168 | 0.086 |
| tiled_p2 (CLAHE) | + water-masked CLAHE contrast | 0.416 | 0.223 | 0.199 | 0.090 |
| tiled_p2 (CLAHE, 500 ep) | longer training; early-stopped ep 139, best ep 39 | 0.292 | 0.206 | 0.160 | 0.076 |

**Tiling** (data-side fix) produced the largest single jump: native-resolution 640 tiles
keep vessels at ~8×5 px instead of crushing them to ~3 px under a whole-scene letterbox.
mAP50 0.069 → 0.184.

**P2 head** (architecture-side fix, §3.6.1): a fourth Detect head at stride 4 (P2/4) added
to the GELAN head of yolov9m.yaml — FPN top-down extended to P2, PAN bottom-up back to P5,
Detect over [P2, P3, P4, P5]. Model validated (189 layers, 19.2 M params, 107.3 GFLOPs).
On its own it traded recall for precision rather than helping — pointing to a data, not
architecture, bottleneck.

### 3.2 The preprocessing-contrast finding (central result)
All architectures plateaued at recall ≈ 0.2. Inspection of the validation prediction tiles
showed the cause: **vessels were not visible in the 8-bit images fed to the detector.** The
uint16→8-bit conversion was the bottleneck, not the model.

Quantified evidence (Marseille, B04 vessel pixels): vessel DN **median 10698**, p75 16403,
max 50626 — vessels are *bright*. Yet the global linear stretch used an upper bound of
**26214** (set by bright land/cloud at the 99th percentile), mapping a median vessel to ~8-bit
value 99 — a dim grey on near-black. Roughly half the vessel contrast was discarded in
preprocessing.

Three stretch attempts demonstrated that **no single global linear stretch suffices**:
- high ceiling (hi≈26000): underway ships lost in near-black water;
- low ceiling (hi≈5100): image binarizes — moored ships swallowed into black harbor.
Moored ships (dim, against dark harbor) and underway ships (bright, on open water) require
*opposite* mappings, which a single linear transform cannot provide.

**Resolution: CLAHE** (contrast-limited adaptive histogram equalization, clipLimit 2.5,
8×8 grid) applied per band after a wide percentile pre-clip. CLAHE equalizes contrast
locally, so dark harbor and bright land each receive appropriate mapping in the same image.
Parameters are fixed and deterministic, so the identical transform applies to clean,
degraded, and denoised images at evaluation. Result: recall 0.205 → 0.223, mAP50 0.184 →
0.199 (best run to date), with rich land texture and visible bright vessels.

### 3.3 Band-specific detectability (supports H2)
A per-vessel contrast check (box mean vs surrounding ring) quantifies a physical limit:
- **B04 (red, 10 m native): 23/41** Marseille vessels clearly brighter than surroundings.
- **B11 (SWIR, 20 m resampled to 10 m): 14/41.**
B11 carries substantially less vessel signal — consistent with its coarser native
resolution and wavelength. This is band-specific behavior the study is designed to surface,
now quantified. Implication: a recall ceiling well below 1.0 is partly physical (in B11,
~27/41 vessels are low-contrast and not reliably learnable from a single band), not a
detector or preprocessing deficiency.

---

## 4. Interpretation for the thesis
This session produced a live demonstration of the central argument: a perceptually
reasonable global stretch silently destroyed task-relevant signal, and three successive
architecture changes could not recover what preprocessing had discarded. Only a task-aware
(adaptive) preprocessing step restored detectability. Preprocessing dominated downstream
detection more than detector architecture did — exactly the perceptual-vs-task decoupling
the praxis sets out to show.

---

## 5. Open items
- **Detector:** finalize training budget (500-epoch run pending); remaining §3.6.1
  adaptations — channel attention and GIoU loss — to add after the P2+CLAHE result is set;
  eval-time tiling + cross-seam NMS stitching script.
- **Denoisers:** four learned methods (DnCNN, Restormer, Noise2Void, Blind2Unblind) trained
  on the 3090; these require the training-partition degraded images (the reason all 432 were
  generated).
- **Factorial:** 648 evaluations (54 conditions × 4 eval scenes × 3 realizations), scoring
  distributed across the cluster.
- **Statistics:** BCa bootstrap, Benjamini–Hochberg correction, Nemenyi post-hoc.
- **Documentation debt:** Ch.3 §3.6.2 Suez wording fix; fold degradation/BM3D provenance and
  this log into the appendices.

## 6. Engineering notes (reproducibility lessons)
- With `imagePullPolicy: Never`, an image tag says nothing about contents — verify with
  `ctr run … import bm3d` before trusting a rebuilt image. A stale 0.1-as-0.2 image cost a
  debugging cycle.
- Scripts edited locally must land on the executing node before they take effect; a one-line
  `grep` guard for a signature string (`tile_starts`, `createCLAHE`) catches stale copies
  before a build or training run is wasted on them.
- Run-numbered filenames (`_second_run`, `_third_run`) caused version drift — a patch applied
  to one copy was lost when another was regenerated. **Recommended cure: a single filename
  per script, versioned in git**, eliminating the "which copy is current" failure class.

---

## 5. Detector correction (5 Sep 2026)

Verification of the primary factorial (re-evaluating bm3d/panama/B04/nominal/r0 and
matching against results/factorial_results.csv) showed the 660 rows were produced by
`tiled_baseline` (SHA-1 7664428a, stock yolov9m.pt, 100 epochs), not `tiled_p2` as
Chapter 3 describes.

Timeline (11 Jun 2026):
- 14:53-14:58  tiled_baseline trained on the tiled dataset as it then existed (linear stretch)
- 20:05        detection_tiled/ rebuilt with CLAHE (P1/P99.5 pre-clip, clip 2.5, grid 8), overwriting the earlier build
- 21:50-22:02  tiled_p2 trained on the CLAHE dataset (MLflow run ac81226, SHA-1 ac2f22fa)
- 29 Aug       factorial evaluated tiled_baseline on CLAHE-stretched tiles

The frozen detector therefore saw one 8-bit mapping in training and another at
evaluation, across all 660 conditions. tiled_p2 is train/eval consistent. On the
verification condition tiled_p2 gives mAP50 0.069 vs 0.011 for tiled_baseline on the
identical input.

**Decision:** re-run the primary factorial and the matched arm with tiled_p2. Rationale
is train/eval consistency, decidable from timestamps without reference to outcomes.
Original results retained at results/ for the record; corrected results written to
results_p2/. The pre-registered rule for delta (half mean recoverable headroom) is
unchanged; its value follows from the corrected headroom.

Section 3.9.7 names YOLOv9l, which does not exist in ultralytics 8.4.51 (variants are
t, s, m, c, e). Amended to YOLOv9c. tiled_p2 early-stopped on default patience; Chapter 3
should say "500 epochs, default patience, best checkpoint retained."

---

## 5. Detector correction (5 Sep 2026)

Verification of the primary factorial (re-evaluating bm3d/panama/B04/nominal/r0 and
matching against results/factorial_results.csv) showed the 660 rows were produced by
`tiled_baseline` (SHA-1 7664428a, stock yolov9m.pt, 100 epochs), not `tiled_p2` as
Chapter 3 describes.

Timeline (11 Jun 2026):
- 14:53-14:58  tiled_baseline trained on the tiled dataset as it then existed (linear stretch)
- 20:05        detection_tiled/ rebuilt with CLAHE (P1/P99.5 pre-clip, clip 2.5, grid 8), overwriting the earlier build
- 21:50-22:02  tiled_p2 trained on the CLAHE dataset (MLflow run ac81226, SHA-1 ac2f22fa); early-stopped ep 139, best ep 39
- 29 Aug       factorial evaluated tiled_baseline on CLAHE-stretched tiles

The frozen detector therefore saw one 8-bit mapping in training and another at
evaluation, across all 660 conditions. tiled_p2 is train/eval consistent. On the
verification condition tiled_p2 gives mAP50 0.069 vs 0.011 for tiled_baseline on the
identical input.

**Decision:** re-run the primary factorial and the matched arm with tiled_p2. Rationale
is train/eval consistency, decidable from timestamps without reference to outcomes.
Original results retained at results/ for the record; corrected results written to
results_p2/. The pre-registered rule for delta (half mean recoverable headroom) is
unchanged; its value follows from the corrected headroom.

Section 3.9.7 names YOLOv9l, which does not exist in ultralytics 8.4.51 (variants are
t, s, m, c, e). Amended to YOLOv9c. Note the reused run name `tiled_p2` overwrote the
earlier CLAHE checkpoint (val mAP50 0.199); the 500-ep run on disk (0.160) is the only
one that exists and the one MLflow tracked.

### Detector sensitivity arm (3.9.7), registered 5 Sep 2026 before execution
Detector: yolov9c + P2 head (scripts/yolov9c-p2.yaml), same edit as yolov9m-p2.yaml.
Recipe identical to tiled_p2 (train_p2.py defaults) except backbone yaml and pretrained
checkpoint; batch 16, falling back to batch 8 only if 16 fails on memory (run name then
carries the _b8 suffix). Evaluated over the FULL grid (660 + 144 matched), not a subset,
since evaluation now takes minutes. Analysis identical to 3.8 on results_p2c/. Perceptual
deltas are detector-independent and expected to reproduce results_p2 exactly.
