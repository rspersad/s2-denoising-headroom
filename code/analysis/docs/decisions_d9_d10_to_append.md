# Append to docs/methodological_decisions.md

The following two decisions are added after running `scripts/explode_coco_annotations.py` (May 24, 2026) and inspecting the post-explosion vessel counts and size distributions.

---

## D9 — Annotation Shape Heterogeneity

**Date surfaced:** May 24, 2026
**Decision:** Accept the heterogeneous annotation styles as a known dataset characteristic; rely on bounding-box outputs rather than polygon-exact masks for the praxis methodology.

**Background:**
- After exploding the S2-SHIPS COCO annotations into per-vessel records, inspection of the polygon shapes reveals that annotators used different conventions across scenes
- Many polygons are 4-vertex diamond shapes serving as bbox-proxies (e.g., annotation 92 in Rome contains seven diamond polygons of approximately 4×4 pixels each)
- Other polygons are line segments (e.g., 9 polygons in Marseille are 1.0 × 2.0 pixel line segments, presumably representing thin small craft)
- A few polygons are irregular shapes of various dimensions

**Rationale:**
- The praxis evaluates vessel detection via mAP@0.5 and mAP@0.5:0.95 on bounding boxes (per Chapter 3.7)
- YOLOv9 consumes bounding boxes, not polygon masks
- The bounding-box format produced by the explosion script normalizes across annotation conventions
- The diamond and line-segment polygons all produce valid axis-aligned bounding boxes via the explosion logic

**Implications:**
- The S2-SHIPS segmentation polygons should be understood as functional bbox-proxies, not pixel-accurate vessel outlines
- Studies requiring pixel-accurate vessel masks would need to use the S2-SHIPS label mask PNGs (in `s2ships_labels_mask/`) instead of the COCO segmentation polygons
- For the praxis bounding-box detection methodology, this heterogeneity does not introduce systematic bias since bounding boxes are computed uniformly across all polygon shapes

**Status:** Confirmed (May 24, 2026)

---

## D10 — Final Vessel Count After Filtering

**Date surfaced:** May 24, 2026
**Decision:** Working dataset is 951 per-vessel annotations across 16 scenes.

**Background:**
- The COCO file `coco-1622202804.029818.json` contains 1,243 polygons across 159 grouped annotation records
- After applying the category filter (ship + moored ship, per D3), 983 polygons remain
- After applying the 3×3 pixel minimum bounding-box filter (per Chapter 3.2.3), 951 polygons remain

**Post-filter per-scene vessel counts:**

| Scene | Final vessels | Size-rejected | Note |
|---|---|---|---|
| brest1 | 145 | 1 | |
| marseille | 41 | 24 | High small-vessel content |
| panama | 69 | 0 | Chokepoint |
| portsmouth | 42 | 0 | |
| rome | 57 | 5 | |
| rotterdam1 | 117 | 0 | |
| rotterdam2 | 32 | 0 | |
| rotterdam3 | 49 | 0 | |
| southampton | 65 | 1 | |
| suez1-5 | 5, 5, 5, 8, 15 | 0 | Chokepoint (low density) |
| suez6 | 35 | 0 | Chokepoint (Red Sea exit) |
| toulon | 261 | 1 | Largest single-scene contribution (27% of total) |
| **TOTAL** | **951** | **32** | |

**Notable observations:**
- The 3×3 size filter rejection rate is 3.2% overall, primarily concentrated in Marseille (24 of 32 total rejections)
- The Marseille rejections reflect line-segment annotations of small craft (1.0 × 2.0 pixel polygons) that fall below the praxis detection threshold
- Toulon dominates the dataset with 261 vessels (27.4% of total); train/val/test partitioning must account for this
- The six Suez chokepoint scenes contribute only 73 vessels combined — most Suez scenes have 5-15 vessels each, reflecting the open-channel nature of canal imagery

**Implications:**
- The working dataset size of 951 per-vessel annotations is sufficient for the praxis experimental design
- The Toulon over-representation reinforces the need for careful scene-level partitioning (see D4 — block partitioning of dependent scenes also applies here)
- The Marseille filter pattern provides evidence that the 3×3 threshold is doing meaningful work (excluding annotations below the practical Sentinel-2 detection limit) rather than being a redundant criterion

**Status:** Confirmed (May 24, 2026)

---

## Updates needed to existing docs

After the next git commit, the following updates should be applied:

1. **docs/data_structure.md:** Update the "Vessel Polygon Counts" table to add a "Post-3×3 filter" column showing the 951 final count.

2. **docs/data_structure.md:** Add a "Working Dataset" section noting that `/data/praxis/processed/s2ships_exploded_coco.json` is the authoritative per-vessel annotation file for downstream praxis work.

3. **docs/data_structure.md:** Add a note on annotation shape heterogeneity (D9 findings).
