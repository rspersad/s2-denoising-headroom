#!/usr/bin/env python3
"""
evaluate_factorial.py - Tile-level detection scoring for the denoising factorial
(Chapter 3, Section 3.8).

For each (method x band x severity x realization x eval-scene), this:
  1. reads the image for that condition (the degraded raster for the 'none'
     baseline, or the method's denoised raster otherwise),
  2. rescales it from the harmonized reflectance x 10000 domain back to the
     detector's native reflectance x 65535 domain (the stretch bounds are
     native-domain), so the tiles are constructed identically to YOLO training,
  3. stretches + tiles it with the EXACT functions and constants used to build
     the detection dataset (imported from build_detection_dataset.py),
  4. runs the frozen tiled YOLOv9m detector and computes tile-level mAP@50 and
     mAP@50-95 against the per-tile COCO ground truth, using Ultralytics' own
     val() so the numbers are directly comparable to the tiled_baseline metric,
  5. writes one row per condition to factorial_results.csv.

Methods: 'none' (degraded, no denoising) + any of {bm3d, dncnn, noise2void,
blind2unblind, restormer} whose denoised rasters are present. Missing method
outputs are skipped with a warning, so this runs on whatever exists (e.g. the
'none' baseline and BM3D before the learned-denoiser outputs are generated).

Run from the repo root:
    python3 scripts/evaluate_factorial.py \
        --manifest    /data/praxis/processed/s2ships_manifest.csv \
        --coco        /data/praxis/processed/s2ships_exploded_coco.json \
        --bounds      /data/praxis/detection_tiled/stretch_bounds.json \
        --weights     /data/praxis/models/yolov9m-detection/tiled_baseline/weights/best.pt \
        --degraded    /data/praxis/synthetic-degradations/sentinel2 \
        --denoised    /data/praxis/denoised \
        --out         /data/praxis/results
"""
import argparse, csv, json, os, sys, shutil, tempfile, warnings

import numpy as np
import rasterio
import cv2
from PIL import Image

# --- import the EXACT tiling/stretch logic the detector dataset was built with ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from build_detection_dataset import (
    tile_starts, boxes_in_tile, to_yolo, apply_stretch,
    load_coco, load_manifest, scene_image_id,
    TILE, STRIDE, MIN_BOX, BANDS, PATH_COL,
)

EVAL_SCENES = ["panama", "rome", "brest1", "rotterdam1"]
SEVERITIES = ["nominal", "moderate", "severe"]
LEARNED = ["dncnn", "noise2void", "blind2unblind", "restormer"]
# 'clean' = the undegraded raster (per-scene upper bound, native domain, no rescale).
ALL_METHODS = ["clean", "none", "bm3d"] + LEARNED

# Harmonized degraded/denoised rasters are reflectance x 10000; the detector
# stretch bounds are reflectance x 65535. Rescale before stretching.
NATIVE_QV, CALIB_QV = 65535.0, 10000.0
TO_NATIVE = NATIVE_QV / CALIB_QV


def degraded_path(root, scene, band, sev, r):
    return os.path.join(root, scene, band, sev, f"r{r}.tif")


def denoised_path(root, method, scene, band, sev, r):
    return os.path.join(root, method, scene, band, sev, f"r{r}.tif")


def read_raster(path):
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.float64)


# --- perceptual metrics (self-contained; no skimage dependency) -------------
def psnr(a, b, data_range=1.0):
    """PSNR between two reflectance images (data_range in reflectance units)."""
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(a, b, data_range=1.0):
    """Global mean SSIM (Wang et al. 2004), 11x11 Gaussian window, via cv2."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    win, sig = (11, 11), 1.5
    mu_a = cv2.GaussianBlur(a, win, sig)
    mu_b = cv2.GaussianBlur(b, win, sig)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, win, sig) - mu_a2
    sb = cv2.GaussianBlur(b * b, win, sig) - mu_b2
    sab = cv2.GaussianBlur(a * b, win, sig) - mu_ab
    smap = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return float(smap.mean())


_CLEAN_REFL = {}  # (scene, band) -> clean reflectance image, cached


def clean_reflectance(manifest_row, band):
    """Clean reference as reflectance in ~[0,1]: native DN / NATIVE_QV."""
    key = (manifest_row["scene_id"], band)
    if key not in _CLEAN_REFL:
        _CLEAN_REFL[key] = read_raster(manifest_row[PATH_COL[band]]) / NATIVE_QV
    return _CLEAN_REFL[key]


def build_tiles_for_condition(arr_native, boxes, band, bounds, tmp):
    """Stretch + tile one NATIVE-domain (reflectance x 65535) scene into a YOLO
    val folder. Returns (n_tiles, n_gt_boxes). The caller is responsible for
    converting harmonized (x 10000) rasters to native before calling."""
    lo, hi = bounds[band]["lo"], bounds[band]["hi"]
    gray = apply_stretch(arr_native, lo, hi)           # uint8, CLAHE
    rgb = np.stack([gray, gray, gray], axis=-1)        # replicate to 3ch
    H, W = gray.shape

    img_dir = os.path.join(tmp, "images", "val")
    lbl_dir = os.path.join(tmp, "labels", "val")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    n_tiles, n_gt = 0, 0
    for y0 in tile_starts(H):
        for x0 in tile_starts(W):
            tb = boxes_in_tile(boxes, x0, y0)
            tile = rgb[y0:y0 + TILE, x0:x0 + TILE]
            # pad if a clamped edge tile is short (keeps 640x640)
            if tile.shape[0] != TILE or tile.shape[1] != TILE:
                pad = np.zeros((TILE, TILE, 3), np.uint8)
                pad[:tile.shape[0], :tile.shape[1]] = tile
                tile = pad
            stem = f"{x0}_{y0}"
            Image.fromarray(tile).save(os.path.join(img_dir, stem + ".png"))
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                for b in tb:
                    xc, yc, w, h = to_yolo(b)
                    f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            n_tiles += 1
            n_gt += len(tb)
    return n_tiles, n_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--bounds", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--degraded", required=True)
    ap.add_argument("--denoised", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--bands", nargs="+", default=BANDS)
    ap.add_argument("--severities", nargs="+", default=SEVERITIES)
    ap.add_argument("--scenes", nargs="+", default=EVAL_SCENES)
    ap.add_argument("--realizations", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    manifest = load_manifest(args.manifest)
    fname_by_id, boxes_by_id = load_coco(args.coco)
    bounds = json.load(open(args.bounds))["bounds"]
    model = YOLO(args.weights)

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "factorial_results.csv")
    fields = ["method", "band", "severity", "realization", "scene",
              "n_tiles", "n_gt", "map50", "map50_95", "psnr", "ssim"]
    written = 0
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()

        for method in args.methods:
            for scene in args.scenes:
                row = manifest[scene]
                iid = scene_image_id(row, fname_by_id)
                if iid is None:
                    print(f"[skip] no COCO match for scene {scene}", flush=True); continue
                boxes = boxes_by_id.get(iid, [])
                for band in args.bands:
                    for sev in args.severities:
                        for r in range(args.realizations):
                            if method == "clean":
                                # clean is identical across severity/realization; score once
                                if sev != args.severities[0] or r != 0:
                                    continue
                                src = row[PATH_COL[band]]
                                if not os.path.exists(src):
                                    print(f"[skip] missing clean raster {scene}/{band}", flush=True)
                                    continue
                                arr_native = read_raster(src)            # already native x65535
                                rec_sev, rec_r = "clean", 0
                            else:
                                if method == "none":
                                    src = degraded_path(args.degraded, scene, band, sev, r)
                                else:
                                    src = denoised_path(args.denoised, method, scene, band, sev, r)
                                if not os.path.exists(src):
                                    print(f"[skip] missing {method}/{scene}/{band}/{sev}/r{r}", flush=True)
                                    continue
                                arr_native = read_raster(src) * TO_NATIVE  # x10000 -> native
                                rec_sev, rec_r = sev, r

                            tmp = tempfile.mkdtemp(prefix="facteval_")
                            try:
                                n_tiles, n_gt = build_tiles_for_condition(arr_native, boxes, band, bounds, tmp)
                                data_yaml = os.path.join(tmp, "data.yaml")
                                with open(data_yaml, "w") as y:
                                    y.write(f"path: {tmp}\ntrain: images/val\nval: images/val\nnames:\n  0: vessel\n")
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    res = model.val(data=data_yaml, imgsz=TILE, conf=args.conf,
                                                    iou=args.iou, device=args.device,
                                                    verbose=False, save=False, plots=False, workers=0,
                                                    project=tmp, name="val", exist_ok=True)
                                map50 = float(res.box.map50)
                                map5095 = float(res.box.map)
                            finally:
                                shutil.rmtree(tmp, ignore_errors=True)

                            # perceptual quality vs the clean reference (reflectance domain)
                            cref = clean_reflectance(row, band)
                            cond_refl = arr_native / NATIVE_QV   # native DN -> reflectance
                            if cond_refl.shape != cref.shape:
                                h = min(cond_refl.shape[0], cref.shape[0]); wd = min(cond_refl.shape[1], cref.shape[1])
                                cond_refl, cref_c = cond_refl[:h, :wd], cref[:h, :wd]
                            else:
                                cref_c = cref
                            ps = psnr(cond_refl, cref_c, data_range=1.0)
                            ss = ssim(cond_refl, cref_c, data_range=1.0)

                            w.writerow({"method": method, "band": band, "severity": rec_sev,
                                        "realization": rec_r, "scene": scene,
                                        "n_tiles": n_tiles, "n_gt": n_gt,
                                        "map50": f"{map50:.5f}", "map50_95": f"{map5095:.5f}",
                                        "psnr": ("inf" if ps == float("inf") else f"{ps:.3f}"),
                                        "ssim": f"{ss:.5f}"})
                            fh.flush()
                            written += 1
                            print(f"[ok] {method}/{scene}/{band}/{rec_sev}/r{rec_r}  "
                                  f"tiles={n_tiles} gt={n_gt}  mAP50={map50:.3f} "
                                  f"PSNR={ps:.2f} SSIM={ss:.3f}",
                                  flush=True)

    print(f"\n[total] wrote {written} rows -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
