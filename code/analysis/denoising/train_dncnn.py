"""
Train DnCNN (supervised) for one band at a given severity (§3.5.2).

Usage:
    python -m denoising.train_dncnn --band B04 --severity nominal
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from . import config as C
from .data import SupervisedPairDataset
from .models import build
from .train_common import train


def loss_step(model, batch, device):
    noisy, clean = batch
    noisy, clean = noisy.to(device), clean.to(device)
    out = model(noisy)
    return F.mse_loss(out, clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True, choices=C.BANDS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--severity", default="nominal", choices=C.SEVERITIES,
                    help="Severity of the training data. Default nominal, which\n"
                         "reproduces the configuration described in Sections 3.6.2\n"
                         "and 3.6.3. Set to moderate or severe to produce the\n"
                         "severity-matched arm required to test H3 as a\n"
                         "within-method contrast.")
    args = ap.parse_args()

    cfg = C.DNCNN
    band, severity = args.band, args.severity
    train_ds = SupervisedPairDataset(band, severity, patch=cfg.patch,
                                     length=cfg.patches_per_epoch, augment=True)
    val_ds = SupervisedPairDataset(band, severity, patch=cfg.patch,
                                   length=cfg.val_patches, augment=False, seed_tag="val")

    model = build("dncnn")
    out = C.ckpt_dir("dncnn", band, severity)
    train(model, train_ds, cfg, out, loss_step,
          val_ds=val_ds, val_step=loss_step, device=args.device,
          tag=f"dncnn/{band}/{severity}")


if __name__ == "__main__":
    main()
