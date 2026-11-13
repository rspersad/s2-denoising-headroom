#!/usr/bin/env python3
"""
build_detection_dataset.py (tiled) — YOLOv9m detection dataset at native resolution
(Chapter 3, §3.6.2 + small-object tiling adaptation, §3.6.1).

Supersedes the whole-scene letterbox build. Each clean band is stretched uint16 -> uint8,
replicated to 3 channels, and sliced into overlapping TILE x TILE patches at native
resolution so small vessels (median 8x5 px) are preserved rather than shrunk by a
938x1783 -> 640 letterbox. Vessel boxes are assigned to the tile containing their center,
clipped to the tile, and dropped below MIN_BOX px. Positive tiles are all kept; empty
tiles are sampled at --neg-ratio. Keep/drop is decided per tile POSITION and applied to
all three bands so the per-band pooling stays consistent.

Partition (Suez -> training):
  train : suez1-6, toulon, rotterdam3, southampton, portsmouth   (10 scenes)
  val   : marseille, rotterdam2                                    (2 scenes)
  eval  : panama, rome, brest1, rotterdam1   -> excluded (tiled at evaluation time)

stretch_bounds.json and tiles_index.csv are written for reuse / eval-time stitching.
"""
import argparse, csv, json, os, sys
import numpy as np
import rasterio
from PIL import Image
import cv2

BANDS = ["B04", "B05", "B11"]
PATH_COL = {"B04": "path_B04", "B05": "path_B05", "B11": "path_B11"}
TRAIN_SCENES = ["suez1","suez2","suez3","suez4","suez5","suez6",
                "toulon","rotterdam3","southampton","portsmouth"]
VAL_SCENES   = ["marseille","rotterdam2"]
EVAL_SCENES  = ["panama","rome","brest1","rotterdam1"]
P_LO, P_HI = 1.0, 99.5            # wide pre-clip (outlier removal) feeding CLAHE
CLAHE_CLIP, CLAHE_GRID = 2.5, 8   # contrast-limited adaptive histogram equalization
SAMPLE_PER_SCENE = 200_000
TILE, STRIDE, MIN_BOX = 640, 512, 3.0


# ---------- pure, unit-testable helpers ----------
def tile_starts(length, tile=TILE, stride=STRIDE):
    """Tile origin positions along one axis; last tile clamped to the edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def boxes_in_tile(boxes, x0, y0, tile=TILE, min_box=MIN_BOX):
    """COCO [x,y,w,h] boxes -> tile-local [x,y,w,h], by center-in-tile then clip."""
    out = []
    for x, y, w, h in boxes:
        cx, cy = x + w / 2.0, y + h / 2.0
        if not (x0 <= cx < x0 + tile and y0 <= cy < y0 + tile):
            continue
        nx0, ny0 = max(x, x0), max(y, y0)
        nx1, ny1 = min(x + w, x0 + tile), min(y + h, y0 + tile)
        nw, nh = nx1 - nx0, ny1 - ny0
        if nw >= min_box and nh >= min_box:
            out.append((nx0 - x0, ny0 - y0, nw, nh))
    return out


def to_yolo(box, tile=TILE):
    """tile-local [x,y,w,h] -> normalized YOLO (xc,yc,w,h)."""
    x, y, w, h = box
    return ((x + w / 2.0) / tile, (y + h / 2.0) / tile, w / tile, h / tile)


def apply_stretch(arr, lo, hi):
    """uint16 band -> uint8 via wide percentile pre-clip then CLAHE local contrast.
    A single global linear stretch cannot serve both moored ships (dim, in dark harbor)
    and underway ships (bright, on open water); CLAHE equalizes contrast locally so both
    separate from their surroundings. Deterministic given fixed clip/grid, so the same
    transform applies identically to clean/degraded/denoised images at evaluation."""
    x = np.clip(arr.astype(np.float64), lo, hi)
    x = ((x - lo) / max(hi - lo, 1e-6) * 65535.0).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    return (clahe.apply(x) / 256).astype(np.uint8)


# ---------- I/O ----------
def load_coco(path):
    d = json.load(open(path))
    fname_by_id = {im["id"]: im["file_name"] for im in d["images"]}
    boxes_by_id = {}
    for a in d["annotations"]:
        boxes_by_id.setdefault(a["image_id"], []).append(a["bbox"])
    return fname_by_id, boxes_by_id


def load_manifest(path):
    return {r["scene_id"]: r for r in csv.DictReader(open(path, newline=""))}


def scene_image_id(row, fname_by_id):
    target = row["coco_image_filename"]
    for iid, fn in fname_by_id.items():
        if fn == target or os.path.basename(fn) == os.path.basename(target):
            return iid
    return None


def compute_stretch(manifest, scenes):
    """Per-band percentile bounds from WATER pixels only (path_water_mask), excluding
    bright land/cloud/rooftops that otherwise inflate the upper percentile and crush
    vessels-on-water into the near-black bin. The same bounds are reused on degraded/
    denoised images at evaluation, so the representation stays consistent."""
    bounds, rng = {}, np.random.default_rng(0)
    for band in BANDS:
        pool = []
        for sid in scenes:
            row = manifest[sid]
            with rasterio.open(row[PATH_COL[band]]) as ds:
                arr = ds.read(1).astype(np.float64)
            with rasterio.open(row["path_water_mask"]) as ms:
                mask = ms.read(1)
            if mask.shape != arr.shape:
                h = min(mask.shape[0], arr.shape[0]); w = min(mask.shape[1], arr.shape[1])
                if abs(mask.shape[0]-arr.shape[0]) > 2 or abs(mask.shape[1]-arr.shape[1]) > 2:
                    sys.exit(f"ERROR: water mask {mask.shape} vs band {arr.shape} differ by >2px for {sid}")
                mask, arr = mask[:h, :w], arr[:h, :w]
            water = mask > 0                       # nonzero = water (verify printed fraction)
            vals = arr[water]; vals = vals[np.isfinite(vals)]
            print(f"  [{band}/{sid}] water_frac={water.mean():.2f} n_water={vals.size}", flush=True)
            if vals.size == 0:
                continue
            if vals.size > SAMPLE_PER_SCENE:
                vals = rng.choice(vals, SAMPLE_PER_SCENE, replace=False)
            pool.append(vals)
        lo, hi = np.percentile(np.concatenate(pool), [P_LO, P_HI])
        bounds[band] = {"lo": float(lo), "hi": float(hi)}
        print(f"[stretch] {band}: lo={lo:.1f} hi={hi:.1f}  (water-masked)", flush=True)
    return bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-ratio", type=float, default=1.0,
                    help="empty tiles kept per positive tile (-1 = keep all)")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    fname_by_id, boxes_by_id = load_coco(args.coco)
    for sub in ("images/train","images/val","labels/train","labels/val"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    bounds = compute_stretch(manifest, TRAIN_SCENES)
    json.dump({"bounds": bounds, "pre_clip_pct": [P_LO, P_HI], "clahe_clip": CLAHE_CLIP, "clahe_grid": CLAHE_GRID}, open(os.path.join(args.out, "stretch_bounds.json"), "w"), indent=2)

    rng = np.random.default_rng(42)
    index_rows, totals = [], {"train": [0, 0, 0], "val": [0, 0, 0]}  # tiles, pos, boxes
    for split, scenes in (("train", TRAIN_SCENES), ("val", VAL_SCENES)):
        for sid in scenes:
            row = manifest[sid]
            W, H = int(row["image_width"]), int(row["image_height"])
            iid = scene_image_id(row, fname_by_id)
            if iid is None:
                sys.exit(f"ERROR: no COCO image match for scene '{sid}'")
            boxes = boxes_by_id.get(iid, [])
            # classify every tile position as positive (has boxes) or negative
            positions = [(x0, y0) for y0 in tile_starts(H) for x0 in tile_starts(W)]
            pos, neg = [], []
            for (x0, y0) in positions:
                tb = boxes_in_tile(boxes, x0, y0)
                (pos if tb else neg).append((x0, y0, tb))
            # keep all positives + sampled negatives
            if args.neg_ratio < 0:
                keep_neg = neg
            else:
                k = min(len(neg), int(round(args.neg_ratio * len(pos))))
                idx = rng.choice(len(neg), k, replace=False) if neg and k else []
                keep_neg = [neg[i] for i in idx]
            kept = pos + keep_neg
            # cache stretched bands once per scene
            band_imgs = {}
            for band in BANDS:
                with rasterio.open(row[PATH_COL[band]]) as ds:
                    band_imgs[band] = apply_stretch(ds.read(1), bounds[band]["lo"], bounds[band]["hi"])
            for (x0, y0, tb) in kept:
                lines = []
                for b in tb:
                    xc, yc, wn, hn = to_yolo(b)
                    lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
                for band in BANDS:
                    crop = band_imgs[band][y0:y0+TILE, x0:x0+TILE]
                    stem = f"{sid}_{band}_{x0}_{y0}"
                    Image.fromarray(np.repeat(crop[:, :, None], 3, axis=2), "RGB").save(
                        os.path.join(args.out, f"images/{split}/{stem}.png"))
                    open(os.path.join(args.out, f"labels/{split}/{stem}.txt"), "w").write("\n".join(lines))
                    index_rows.append({"split": split, "scene_id": sid, "band": band,
                                       "x0": x0, "y0": y0, "n_boxes": len(tb),
                                       "image": f"images/{split}/{stem}.png"})
                totals[split][0] += 1
                totals[split][1] += 1 if tb else 0
                totals[split][2] += len(tb)
            print(f"[{split}] {sid}: {len(pos)} pos + {len(keep_neg)} neg tiles "
                  f"x3 bands, {sum(len(t[2]) for t in kept)} boxes/band", flush=True)

    with open(os.path.join(args.out, "tiles_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split","scene_id","band","x0","y0","n_boxes","image"])
        w.writeheader(); w.writerows(index_rows)
    open(os.path.join(args.out, "data.yaml"), "w").write(
        f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: vessel\n")
    for s in ("train","val"):
        t, p, b = totals[s]
        print(f"[total] {s}: {t} tile-images ({p} positive) across 3 bands, {b} boxes")
    print(f"[done] tiled dataset at {args.out}")


if __name__ == "__main__":
    main()
