# Chapter 3 additions surfaced by the handoff

These supplement `Detector_Ladder_Corrections.docx`. Section numbers are the
post-move numbering from Part 0 of that file. Praxis text is set off as
block quotes; everything else is instruction.

## Closes Part 8, item 1: paragraph 717 (now in 3.6.1)

The handoff resolves the contradiction. The m seed-0 run trained for 139
epochs with the best validation epoch at 39. Replace "Training was
terminated at epoch 39, with no further improvement in performance on the
validation set." with:

> Validation performance reached its maximum at epoch 39, and training
> stopped at epoch 139 under the patience rule with no further improvement.

## 3.6.3 Detection Model Training: initialization (new sentence after the paragraph-829 opening)

Chapter 3 never states how the detectors were initialized. The runs
transferred backbone weights from the stock COCO-pretrained checkpoints with
the P2 head and reshaped head layers initialized fresh. The DnCNN/Restormer
distinction you drew earlier requires the same precision here.

> Each detector was initialized from the stock COCO-pretrained checkpoint
> for its scale, with backbone weights transferred and the P2 detection head
> and reshaped output layers initialized fresh, and was then trained on the
> clean training partition.

## 3.6.3: augmentation (replaces "standard mosaic augmentation and random horizontal flipping" in the Part 2 text)

> Augmentation followed the Ultralytics 8.4.51 defaults: mosaic with
> close-mosaic over the final 10 epochs, horizontal flip with probability
> 0.5, scale jitter of 0.5, translation of 0.1, and HSV jitter of 0.015,
> 0.7, and 0.4, with a three-epoch warmup and automatic mixed precision.

## 3.8.7: three sentences the scripts commit you to (append to the Part 4 text)

> Inferential comparisons are made on the seed-0 detector of each variant,
> which for YOLOv9m is the primary detector of Chapter 4. The two further
> seeds per variant supply a descriptive measure of training variability
> and are not pooled as additional blocks. The first criterion is assessed
> both pooled across conditions and within each band-by-severity cell,
> because a capacity-dependent effect confined to a single cell would not
> register in the pooled contrast.

## 3.9 Statistical Computing Environment: reproducibility (replaces the Part 5 placeholder text)

Fill the hash from `git rev-parse --short HEAD` on the branch; the handoff
gives `44f490b`.

> The denoising, detection, and metric-computation pipeline is version
> controlled in the repository `rspersad/praxis`. The runs reported in
> Chapter 4, including the detector sensitivity arms, were produced from the
> branch `correction/detector-tiled-p2` at commit `44f490b` using Ultralytics
> 8.4.51, PyTorch 2.4.0 with CUDA 12.1, and Python 3.11. The YOLOv9c
> architecture definition and the seeded training launcher entered the
> repository at that commit, after the September YOLOv9c seed-0 run, which
> was produced from the same files before they were committed. The commit
> identifier, package versions, and hardware allocation for each run are
> recorded in Appendix D.

## Chapter 4 seed-spread paragraph: hardware (one sentence, wherever the three-seed table is introduced)

> Seed-to-seed spread for the m and c variants includes a hardware
> component, since the three seeds of each were split between an RTX 3090
> and an RTX A4500 and bitwise reproducibility across GPUs is not
> guaranteed; all three e seeds ran on the RTX 3090.

## Section-number hygiene

The handoff cites 3.9.7 and 3.8.7 interchangeably. After the Part 0 move
it is 3.8.7 everywhere. The advisor deck already says 3.8.7.
