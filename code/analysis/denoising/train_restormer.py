"""
Fine-tune Restormer (supervised) for one band at nominal severity (§3.5.3).

Restormer is initialized from the official released grayscale Gaussian-denoising
checkpoint, then fine-tuned 50 epochs on our per-band nominal Poisson-Gaussian
pairs. To keep checkpoint keys compatible we use the AUTHORS' architecture
rather than a reimplementation.

Setup (one-time, on Astraea):
    # vendor the official model definition
    pip install einops
    git clone https://github.com/swz30/Restormer external/Restormer
    # grayscale blind denoising weights -> config.RESTORMER_PRETRAINED
    #   https://github.com/swz30/Restormer/releases  (Gaussian_Gray_Denoising)

Usage:
    python -m denoising.train_restormer --band B04 \
        --arch external/Restormer/basicsr/models/archs/restormer_arch.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from . import config as C
from .data import SupervisedPairDataset
from .train_common import train


def load_restormer_arch(arch_path: str):
    """Import the official Restormer class from a vendored arch file."""
    p = Path(arch_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Restormer arch not found at {arch_path}. Clone swz30/Restormer "
            "(see module docstring) and pass --arch to its restormer_arch.py."
        )
    spec = importlib.util.spec_from_file_location("restormer_arch", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["restormer_arch"] = mod
    spec.loader.exec_module(mod)
    return mod.Restormer


def build_restormer(arch_path: str):
    Restormer = load_restormer_arch(arch_path)
    # Single-channel config matching the grayscale denoising checkpoint.
    model = Restormer(
        inp_channels=1, out_channels=1, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False,
    )
    ckpt_path = C.RESTORMER_PRETRAINED
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("params", sd)  # official checkpoints nest under 'params'
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded Restormer pretrained; missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"WARNING: pretrained checkpoint {ckpt_path} not found; training from scratch.")
    return model


def loss_step(model, batch, device):
    noisy, clean = batch
    noisy, clean = noisy.to(device), clean.to(device)
    out = model(noisy)
    # L1 is the Restormer default restoration loss.
    return F.l1_loss(out, clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True, choices=C.BANDS)
    ap.add_argument("--arch", required=True,
                    help="Path to official restormer_arch.py")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--severity", default="nominal", choices=C.SEVERITIES,
                    help="Severity of the training data. Default nominal, which\n"
                         "reproduces the configuration described in Sections 3.6.2\n"
                         "and 3.6.3. Set to moderate or severe to produce the\n"
                         "severity-matched arm required to test H3 as a\n"
                         "within-method contrast.")
    args = ap.parse_args()

    cfg = C.RESTORMER
    band, severity = args.band, args.severity
    train_ds = SupervisedPairDataset(band, severity, patch=cfg.patch,
                                     length=cfg.patches_per_epoch, augment=True)
    val_ds = SupervisedPairDataset(band, severity, patch=cfg.patch,
                                   length=cfg.val_patches, augment=False, seed_tag="val")

    model = build_restormer(args.arch)
    out = C.ckpt_dir("restormer", band, severity)
    train(model, train_ds, cfg, out, loss_step,
          val_ds=val_ds, val_step=loss_step, device=args.device,
          tag=f"restormer/{band}/{severity}")


if __name__ == "__main__":
    main()
