"""
Train Blind2Unblind (self-supervised) for one band at one severity (§3.5.5).

This implements the Blind2Unblind scheme in the project torch stack: a global
cell-mask blind-spot term plus the "re-visible" term that lets the network use
the masked pixel's own context (Wang et al., 2022). A lambda ramp weights the
re-visible term over training.

>>> FIDELITY NOTE <<<
Chapter 3 §3.5.5 specifies "the reference implementation distributed by the
authors." This file is a faithful structural reimplementation, not the literal
author repo. B2U's re-visible loss is the subtlest piece in the whole pipeline,
so this is the one method worth either (a) validating numerically against the
authors' code, or (b) swapping in directly. README has the swap-in recipe. The
re-visible weighting and masker geometry below are the knobs to check.

Usage:
    python -m denoising.train_blind2unblind --band B04 --severity severe
"""
from __future__ import annotations

import argparse
import random

import torch

from . import config as C
from .data import NoisyOnlyDataset
from .masking import b2u_cell_mask, b2u_offsets, replace_with_neighbour
from .models import build
from .train_common import train

CELL = 2           # global-masker cell size (masks 1/CELL^2 of pixels per step)
LAMBDA_MAX = 2.0   # re-visible term weight (ramped)
RAMP_EPOCHS = 20   # epochs over which lambda ramps from 0 -> LAMBDA_MAX


def _lambda_for_epoch(model) -> float:
    ep = getattr(model, "_b2u_epoch", 0)
    return LAMBDA_MAX * min(1.0, ep / max(1, RAMP_EPOCHS))


def loss_step(model, batch, device):
    noisy = batch.to(device)
    offset = random.choice(b2u_offsets(CELL))
    mask = b2u_cell_mask(noisy, CELL, offset)

    # Blind term: predict masked pixels from a neighbour-replaced input.
    masked_in = replace_with_neighbour(noisy, mask)
    blind_out = model(masked_in)
    denom = mask.sum().clamp(min=1.0)
    blind_loss = (((blind_out - noisy) ** 2) * mask).sum() / denom

    # Re-visible term: direct (unmasked) prediction provides extra context;
    # detached so it acts as a target that pulls the blind output toward the
    # information-richer estimate without collapsing to identity.
    with torch.no_grad():
        revisible_target = model(noisy)
    lam = _lambda_for_epoch(model)
    rev_loss = (((blind_out - revisible_target) ** 2) * mask).sum() / denom

    return blind_loss + lam * rev_loss


class _EpochTagger:
    """Tiny hook so the loss can see the current epoch for the lambda ramp."""
    def __init__(self, model):
        self.model = model

    def __call__(self, epoch):
        self.model._b2u_epoch = epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True, choices=C.BANDS)
    ap.add_argument("--severity", required=True, choices=C.SEVERITIES)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = C.BLIND2UNBLIND
    band, severity = args.band, args.severity
    train_ds = NoisyOnlyDataset(band, severity, patch=cfg.patch,
                                length=cfg.patches_per_epoch, augment=True)
    val_ds = NoisyOnlyDataset(band, severity, patch=cfg.patch,
                              length=cfg.val_patches, augment=False, seed_tag="val")

    model = build("blind2unblind")
    model._b2u_epoch = 0
    out = C.ckpt_dir("blind2unblind", band, severity)

    # The generic loop doesn't expose an epoch callback, so we ramp lambda by
    # reading model._b2u_epoch; update it via a lightweight monkey-patch on the
    # dataset length boundary. Simpler: bump it once per epoch here by wrapping
    # train() with a callback is overkill -- instead we set RAMP_EPOCHS small and
    # advance the counter from the train loop through the model attribute below.
    from . import train_common
    orig_cosine = train_common.cosine_lr

    def cosine_with_epoch(optimizer, base_lr, epoch, total_epochs, warmup=2):
        model._b2u_epoch = epoch
        return orig_cosine(optimizer, base_lr, epoch, total_epochs, warmup)

    train_common.cosine_lr = cosine_with_epoch
    try:
        train(model, train_ds, cfg, out, loss_step,
              val_ds=val_ds, val_step=loss_step, device=args.device,
              tag=f"b2u/{band}/{severity}")
    finally:
        train_common.cosine_lr = orig_cosine


if __name__ == "__main__":
    main()
