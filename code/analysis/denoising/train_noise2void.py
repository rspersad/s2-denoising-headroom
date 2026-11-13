"""
Train Noise2Void (self-supervised) for one band at one severity (§3.5.4).

Faithful reimplementation of the N2V blind-spot scheme in the project's torch
stack: random pixels are masked via neighbour-replacement and the MSE loss is
computed only at the masked locations. See README "Fidelity note" -- Chapter 3
cites the n2v library by name; if you prefer the literal library, see the
swap-in instructions there.

Usage:
    python -m denoising.train_noise2void --band B04 --severity moderate
"""
from __future__ import annotations

import argparse

import torch

from . import config as C
from .data import NoisyOnlyDataset
from .masking import n2v_mask
from .models import build
from .train_common import train


def make_loss_step(rate: float):
    def loss_step(model, batch, device):
        noisy = batch.to(device)
        masked, mask = n2v_mask(noisy, rate)
        pred = model(masked)
        # Loss only at blind-spot pixels; target is the original noisy value.
        diff = (pred - noisy) ** 2 * mask
        denom = mask.sum().clamp(min=1.0)
        return diff.sum() / denom
    return loss_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True, choices=C.BANDS)
    ap.add_argument("--severity", required=True, choices=C.SEVERITIES)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = C.NOISE2VOID
    band, severity = args.band, args.severity
    train_ds = NoisyOnlyDataset(band, severity, patch=cfg.patch,
                                length=cfg.patches_per_epoch, augment=True)
    val_ds = NoisyOnlyDataset(band, severity, patch=cfg.patch,
                              length=cfg.val_patches, augment=False, seed_tag="val")

    model = build("noise2void")
    step = make_loss_step(cfg.mask_rate)
    out = C.ckpt_dir("noise2void", band, severity)
    train(model, train_ds, cfg, out, step,
          val_ds=val_ds, val_step=step, device=args.device,
          tag=f"n2v/{band}/{severity}")


if __name__ == "__main__":
    main()
