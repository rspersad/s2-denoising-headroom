#!/usr/bin/env python3
"""
denoise_eval.py - Apply a trained learned denoiser to the degraded EVALUATION
images, writing denoised rasters in the same layout BM3D uses so the scoring
harness (evaluate_factorial.py) reads every method uniformly.

For each eval scene x band x severity x realization, this loads the checkpoint
matching that (band, severity) -- /data/praxis/models/denoisers/{method}/{band}_{sev}/best.pth
-- and denoises the degraded raster, writing
   /data/praxis/denoised/{method}/{scene}/{band}/{sev}/r{N}.tif

Domain: the degraded rasters are harmonized (reflectance x 10000). Training fed
the model harmonized/REFLECTANCE_SCALE (reflectance in ~[0,1]); inference mirrors
that exactly: raster -> /REFLECTANCE_SCALE -> model -> xREFLECTANCE_SCALE -> raster.
Output stays in the harmonized domain, identical to BM3D's outputs.

Inference is overlap-tiled (reflection-padded to a multiple of PAD) and the
overlaps are averaged, so it is seam-free and fits any of the four architectures
(DnCNN, U-Net, Restormer) within GPU memory.

Conditions whose (band, severity) checkpoint is absent are skipped (e.g. DnCNN /
Restormer were trained at nominal only).

Run from the repo root, e.g.:
    python3 scripts/denoise_eval.py --method noise2void \
        --degraded /data/praxis/synthetic-degradations/sentinel2 \
        --out /data/praxis/denoised \
        --ckpt-root /data/praxis/models/denoisers
"""
import argparse, os, sys
import numpy as np
import rasterio
import torch

sys.path.insert(0, os.getcwd())
from denoising import config as C
from denoising.models import build as build_simple

EVAL_SCENES = ["panama", "rome", "brest1", "rotterdam1"]
BANDS = ["B04", "B05", "B11"]
SEVERITIES = ["nominal", "moderate", "severe"]
PAD = 16          # pad tiles to a multiple of this (U-Net depth / Restormer windows)


def load_model(method, ckpt_path, device, arch_path=None):
    if method == "restormer":
        sys.path.insert(0, os.path.dirname(arch_path))
        import importlib.util
        spec = importlib.util.spec_from_file_location("restormer_arch", arch_path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        model = mod.Restormer(
            inp_channels=1, out_channels=1, dim=48,
            num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
            heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False,
        )
    else:
        model = build_simple(method)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.no_grad()
def denoise_tiled(model, img, device, tile=256, overlap=32):
    """Overlap-tiled inference on a single-channel float image in [0,1]-ish.
    Reflection-pads each tile to a multiple of PAD; averages overlaps."""
    H, W = img.shape
    step = tile - overlap
    acc = np.zeros((H, W), np.float64)
    wsum = np.zeros((H, W), np.float64)
    ys = list(range(0, max(1, H - overlap), step))
    xs = list(range(0, max(1, W - overlap), step))
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            y0c, x0c = max(0, y1 - tile), max(0, x1 - tile)
            patch = img[y0c:y1, x0c:x1]
            ph, pw = patch.shape
            Ph = (ph + PAD - 1) // PAD * PAD
            Pw = (pw + PAD - 1) // PAD * PAD
            pad = np.pad(patch, ((0, Ph - ph), (0, Pw - pw)), mode="reflect")
            t = torch.from_numpy(pad).float()[None, None].to(device)
            out = model(t)[0, 0].clamp_min(0).cpu().numpy()[:ph, :pw]
            acc[y0c:y1, x0c:x1] += out
            wsum[y0c:y1, x0c:x1] += 1.0
    wsum[wsum == 0] = 1.0
    return acc / wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["dncnn", "noise2void", "blind2unblind", "restormer"])
    ap.add_argument("--degraded", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-root", default="/data/praxis/models/denoisers")
    ap.add_argument("--arch", default="external/Restormer/basicsr/models/archs/restormer_arch.py")
    ap.add_argument("--scenes", nargs="+", default=EVAL_SCENES)
    ap.add_argument("--bands", nargs="+", default=BANDS)
    ap.add_argument("--severities", nargs="+", default=SEVERITIES)
    ap.add_argument("--realizations", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--train-severity", default=None,
                    help="Severity of the checkpoint to load, regardless of the "
                         "severity being evaluated. The supervised methods are "
                         "trained at nominal and evaluated across all severities "
                         "(Section 3.6.2). Default: match the evaluation severity, "
                         "as the self-supervised methods require.")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    scale = float(C.REFLECTANCE_SCALE)
    cache = {}                       # (band,sev) -> model, loaded lazily
    written, skipped = 0, 0

    for band in args.bands:
        for sev in args.severities:
            ckpt_sev = args.train_severity or sev
            ckpt = os.path.join(args.ckpt_root, args.method, f"{band}_{ckpt_sev}", "best.pth")
            if not os.path.exists(ckpt):
                print(f"[skip] no checkpoint for {args.method}/{band}_{ckpt_sev}", flush=True)
                continue
            model = load_model(args.method, ckpt, device, arch_path=args.arch)
            print(f"[load] {args.method}/{band}_{ckpt_sev} -> eval {sev}", flush=True)
            for scene in args.scenes:
                for r in range(args.realizations):
                    src = os.path.join(args.degraded, scene, band, sev, f"r{r}.tif")
                    if not os.path.exists(src):
                        print(f"[skip] missing degraded {scene}/{band}/{sev}/r{r}", flush=True)
                        skipped += 1; continue
                    with rasterio.open(src) as ds:
                        arr = ds.read(1).astype(np.float64)
                        profile = ds.profile.copy()
                    den = denoise_tiled(model, arr / scale, device, tile=args.tile) * scale
                    dtype = profile["dtype"]
                    if np.issubdtype(np.dtype(dtype), np.integer):
                        den = np.clip(np.rint(den), 0, np.iinfo(dtype).max)
                    den = den.astype(dtype)
                    out_dir = os.path.join(args.out, args.method, scene, band, sev)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"r{r}.tif")
                    with rasterio.open(out_path, "w", **profile) as dst:
                        dst.write(den, 1)
                    written += 1
                    print(f"[ok] {args.method}/{scene}/{band}/{sev}/r{r}", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"\n[total] {args.method}: wrote {written}, skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
