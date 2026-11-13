"""Score every annotated vessel as matched / unmatched per condition from the predictions saved by
evaluate_factorial.py --save-pred.  One row per vessel x method x band x severity x realization.

Usage: python3 -I score_vessels.py --pred /data/praxis/results_vessel/pred --out /data/praxis/results_vessel/vessels.csv [--iou 0.5] [--tile 640]

Matching: within each tile, predictions are taken in descending confidence and each is matched to the
unmatched ground-truth box with the highest IoU at or above --iou (greedy, the usual detection rule).
A vessel's score is the confidence of the prediction matched to it, 0 if none.  Vessels that appear in
more than one overlapping tile (including boxes clipped at a tile edge) are merged when their scene-coordinate
boxes overlap by more than half of the smaller box; the merged vessel takes the maximum score and the size of
its least-clipped box.
"""
import argparse, os, json, re, csv, glob
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--iou", type=float, default=0.5); ap.add_argument("--tile", type=int, default=640)
args = ap.parse_args()

def iou_matrix(a, b):  # a: (n,4) xyxy, b: (m,4)
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)

def tile_offsets(stems):
    """Return {stem: (x_off, y_off)}.  Stems end in two integers; the position with the larger range is x (scene is wider than tall)."""
    nums = {s: tuple(int(v) for v in re.findall(r"(\d+)", s)[-2:]) for s in stems}
    a_max = max(v[0] for v in nums.values()); b_max = max(v[1] for v in nums.values())
    first_is_x = a_max >= b_max
    return {s: ((v[0], v[1]) if first_is_x else (v[1], v[0])) for s, v in nums.items()}

rows = []
conds = sorted(glob.glob(os.path.join(args.pred, "*", "*", "*", "*")))
for cd in conds:
    parts = cd.split(os.sep); method, scene, band, sr = parts[-4], parts[-3], parts[-2], parts[-1]
    sev, r = sr.rsplit("_r", 1); r = int(r)
    pj = os.path.join(cd, "predictions.json")
    preds = json.load(open(pj)) if os.path.exists(pj) else []
    by_img = {}
    for p in preds: by_img.setdefault(str(p["image_id"]), []).append(p)
    stems = [os.path.splitext(l.strip())[0] for l in open(os.path.join(cd, "tiles.txt")) if l.strip()]
    offs = tile_offsets(stems)
    vessels = {}  # scene-box key -> dict
    boxes = []    # [x1, y1, x2, y2, matched_conf] in scene coordinates, one per GT box per tile
    for stem in stems:
        lab = os.path.join(cd, "labels", stem + ".txt")
        gt = []
        if os.path.exists(lab):
            for line in open(lab):
                f = line.split()
                if len(f) < 5: continue
                cx, cy, w, h = (float(v) * args.tile for v in f[1:5])
                gt.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        gt = np.array(gt, float).reshape(-1, 4)
        pr = sorted(by_img.get(stem, []), key=lambda p: -p["score"])
        pb = np.array([[p["bbox"][0], p["bbox"][1], p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]] for p in pr], float).reshape(-1, 4)
        sc = np.array([p["score"] for p in pr], float)
        matched = np.zeros(len(gt), float); taken = np.zeros(len(gt), bool)
        M = iou_matrix(pb, gt)
        for i in range(len(pb)):
            if len(gt) == 0: break
            cand = np.where(~taken)[0]
            if len(cand) == 0: break
            j = cand[np.argmax(M[i, cand])]
            if M[i, j] >= args.iou: taken[j] = True; matched[j] = sc[i]
        xo, yo = offs[stem]
        for j, g in enumerate(gt):
            boxes.append([g[0] + xo, g[1] + yo, g[2] + xo, g[3] + yo, matched[j]])
    # merge the same vessel seen in overlapping tiles: boxes whose intersection covers more than half of the smaller box
    B = np.array(boxes, float).reshape(-1, 5)
    parent = list(range(len(B)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    if len(B):
        ix1 = np.maximum(B[:, None, 0], B[None, :, 0]); iy1 = np.maximum(B[:, None, 1], B[None, :, 1])
        ix2 = np.minimum(B[:, None, 2], B[None, :, 2]); iy2 = np.minimum(B[:, None, 3], B[None, :, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]); mn = np.minimum(area[:, None], area[None, :])
        ov = inter / (mn + 1e-9)
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                if ov[i, j] > 0.5: parent[find(i)] = find(j)
    groups = {}
    for i in range(len(B)): groups.setdefault(find(i), []).append(i)
    for members in groups.values():
        g = B[members]; big = g[np.argmax((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))]  # least-clipped box carries the size
        key = (scene, int(round(big[0])), int(round(big[1])), int(round(big[2])), int(round(big[3])))
        vessels[key] = {"conf": float(g[:, 4].max()), "n_tiles": len(members), "w": big[2] - big[0], "h": big[3] - big[1]}
    for key, v in vessels.items():
        rows.append(dict(method=method, scene=scene, band=band, severity=sev, realization=r,
                         vessel_id=f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}", x1=key[1], y1=key[2], x2=key[3], y2=key[4],
                         w=round(v["w"], 2), h=round(v["h"], 2), n_tiles=v["n_tiles"], conf=round(v["conf"], 5)))
    print(f"{method:6s} {scene:11s} {band} {sev:8s} r{r}: {len(vessels)} vessels, {sum(1 for v in vessels.values() if v['conf'] > 0)} matched at any confidence", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)
print("wrote", args.out, len(rows), "rows")
