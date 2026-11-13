# Methodological Decisions Log

This document records methodological decisions made during data acquisition, processing, and experimental design. Decisions are recorded with their date, the option chosen, the rationale, and any open questions requiring praxis director adjudication.

---

## Format

Each decision is recorded with the following structure:

- **ID and topic**
- **Date decided / surfaced**
- **Decision** (or "Open — pending advisor input")
- **Options considered**
- **Rationale**
- **Implications**
- **Status:** Confirmed / Tentative / Open

---

## D1 — Vessel Detection Benchmark Selection

**Date surfaced:** May 24, 2026
**Decision:** Use S2-SHIPS (Ciocarlan & Stoian 2021) as the sole vessel annotation benchmark.

**Options considered:**
- S2-SHIPS alone — peer-reviewed, ESA-funded, covers Suez + Panama chokepoints, ~983 usable vessels
- Sentinel-2-Ships (AGENIUM/CORTEX) alone — Danish waters only, no chokepoint coverage
- MMShip — no length/width annotations, limits size filtering
- mayrajeo Finnish coast dataset — heavy recreational craft, vessel size mismatch with praxis methodology
- S2-SHIPS + Sentinel-2-Ships combined — adds Danish waters but does not address chokepoint gap

**Rationale:**
- S2-SHIPS is the most-cited Sentinel-2 ship detection benchmark in the literature
- Provides direct experimental coverage of 2 of the 6 named chokepoints (Suez, Panama)
- Vessel size distribution aligns with the praxis 3×3 pixel (~30m) minimum threshold
- Peer-reviewed publication and European Space Agency funding provide methodological authority
- The combined-benchmark configuration was rejected because Sentinel-2-Ships (Danish waters) does not address the geographic gap that S2-SHIPS-alone honestly acknowledges

**Implications:**
- Section 1.8 (Research Limitations) must acknowledge that direct experimental coverage of Hormuz, Malacca, Bab-el-Mandeb, and Gibraltar is outside the benchmark scope
- Section 3.2.3 (Vessel Annotations) names S2-SHIPS specifically with citation
- Transferability of conclusions to non-covered chokepoints is grounded in the physical uniformity of Sentinel-2 sensor noise characteristics across geographies

**Status:** Confirmed (May 24, 2026)

---

## D2 — Sentinel-2 Product Level (L1C vs L2A)

**Date surfaced:** May 24, 2026
**Decision:** Open — pending advisor input.

**Background:**
- Chapter 3 v2 currently specifies "Sentinel-2 Level-1C top-of-atmosphere reflectance products"
- The S2-SHIPS distribution provides Sentinel-2 L2A (bottom-of-atmosphere, atmospherically corrected via SEN2COR) products
- The Drusch et al. (2012) and Gorroño et al. (2017) sensor noise characterizations cited in Chapter 2.6 apply directly to L1C products; L2A noise characteristics differ because atmospheric path radiance has been removed

**Options:**
- (a) Update Chapter 3 to specify L2A; accept and document the data we have
- (b) Acquire native L1C versions of the same scenes from the Copernicus Data Space Ecosystem; treat S2-SHIPS as annotation source only
- (c) Use the S2-SHIPS provided L2A imagery but rederive the noise model parameters from L2A characterization (additional research)

**Tentative direction:**
Option (a) appears most pragmatic for the praxis timeline. The Poisson-Gaussian noise model still applies to L2A products with adjusted parameters; the methodology can characterize noise in L2A space rather than L1C space without invalidating the experimental design.

**Status:** Open — flagged for praxis advisor discussion.

---

## D3 — Vessel Category Selection

**Date surfaced:** May 24, 2026
**Decision:** Use `ship` + `moored ship` as positive class; exclude `marina` and `doubt`.

**Background:**
- The canonical COCO file (`coco-1622202804.029818.json`) defines 4 categories: ship, marina, doubt, moored ship
- The praxis methodology evaluates detection performance, not classification — but the categories still require handling

**Options considered:**
- All 4 categories combined as "vessel" — most inclusive, but includes marina clusters and low-confidence annotations
- `ship` only — most restrictive
- `ship` + `moored ship`, exclude `marina` + `doubt` — middle ground (chosen)
- Keep categories separate as multi-class detection — changes the problem from detection to classification

**Rationale:**
- `marina` polygons represent dense recreational craft clusters dominated by small (<30m) recreational boats, which would not survive the 3×3 pixel filter anyway
- `doubt` polygons represent annotator uncertainty — including them introduces label noise
- The moored vs. underway distinction is not reliably applied across scenes (e.g., Portsmouth has 42 ship annotations and 0 moored, suggesting different annotation conventions)
- Combining ship + moored ship into a single "vessel" class is consistent with the praxis goal of detection (presence) rather than classification (state)
- 983 usable vessels after this filter is sufficient for the experimental design

**Implications:**
- Section 3.2.3 (Vessel Annotations) describes this filter explicitly
- The exploded per-vessel annotation file produced by `scripts/explode_coco_annotations.py` (to be created) will apply this filter

**Status:** Tentative — confirm with advisor before code freeze.

---

## D4 — Treatment of Six Suez Scenes (Spatial Dependence)

**Date surfaced:** May 24, 2026
**Decision:** Open — pending advisor input.

**Background:**
- Six of the 16 S2-SHIPS scenes are labeled `suez1` through `suez6`
- All six have acquisition date 2021-04-28
- EPSG:3857 bounds inspection confirms they are spatially adjacent tiles along the Suez Canal (from north entrance at ~30.85°N down to ~29.6°N near the Red Sea exit)
- `suez1` and `suez2` have an approximately 9 km overlap
- They are NOT independent acquisitions; they are sub-tiles of a single Sentinel-2 overpass

**Why this matters:**
- The Wilcoxon signed-rank and Friedman tests specified in Chapter 3.8 assume statistical independence of paired observations
- Treating six dependent sub-tiles as six independent observations would inflate the effective sample size and bias significance test results
- For chokepoint reporting, this is effectively 1 chokepoint acquisition (Suez, sub-tiled) plus 1 separate acquisition (Panama) — not 7 chokepoint scenes

**Options:**
- (a) Treat as six independent observations with documented spatial dependence
- (b) Treat as one acquisition with six spatial sub-samples; apply cluster-robust statistical methods or block-bootstrap
- (c) Consolidate to a single representative tile (loses data but ensures independence)
- (d) Use them all but report sensitivity analyses with each treatment

**Implications regardless of choice:**
- The seven-chokepoint count in the manifest (`is_chokepoint=True` for all 6 Suez + Panama) should be interpreted carefully
- Train/val/test partitioning should keep all six Suez scenes in one partition (block partitioning) to avoid the same-acquisition leakage problem

**Status:** Open — flagged for praxis advisor discussion.

---

## D5 — Native Resolution vs. S2-SHIPS Resampled Bands

**Date surfaced:** May 24, 2026
**Decision:** Open — pending advisor input.

**Background:**
- Native Sentinel-2 resolution: B04 = 10m, B05 = 20m, B11 = 20m
- S2-SHIPS distributes all bands resampled to a common 10m grid (verified via rasterio inspection)
- Chapter 3 v2 specifies "20m red-edge" and "20m SWIR" with the implication that the noise characterization respects native resolution
- The resampling method used by S2-SHIPS is not documented but appears to be either nearest-neighbor or bilinear

**Options:**
- (a) Accept the 10m-resampled bands as provided; document in Chapter 3 that S2-SHIPS distributes resampled imagery
- (b) Re-acquire B05 and B11 at native 20m for the same dates/tiles via the Copernicus Data Space Ecosystem; treat S2-SHIPS as annotation source only
- (c) Resample our own band data uniformly to match the S2-SHIPS distribution, ensuring consistent treatment

**Implications:**
- Option (a) is the simplest and matches operational reality (analysts often work with resampled-to-common-grid products)
- Option (b) is methodologically purer but doubles data acquisition workload
- Option (c) introduces an additional preprocessing step on the praxis side

**Status:** Open — flagged for praxis advisor discussion.

---

## D6 — Coordinate Reference System (EPSG:3857 vs UTM)

**Date surfaced:** May 24, 2026
**Decision:** Accept EPSG:3857 as provided; document in Chapter 3.

**Background:**
- Native Sentinel-2 products are delivered in UTM zone projections appropriate to the tile location
- S2-SHIPS distributes all imagery reprojected to EPSG:3857 (Web Mercator)
- EPSG:3857 is a display projection with latitude-dependent distortion

**Rationale for accepting:**
- The distortion is small at the latitudes covered (most scenes between 30-50°N)
- Re-projecting back to UTM would introduce additional resampling artifacts on top of the S2-SHIPS resampling
- The praxis focus is denoising performance, not absolute spatial measurement accuracy

**Implications:**
- Spatial measurements (vessel size in meters) computed from EPSG:3857 pixel coordinates have a small latitude-dependent error
- For the 3×3 pixel filter (~30m), this error is negligible
- The praxis is not making claims about absolute vessel size from imagery; it uses pixel-size thresholds

**Status:** Confirmed (May 24, 2026)

---

## D7 — Saturated Pixel Handling

**Date surfaced:** May 24, 2026
**Decision:** Mask saturated pixels (value = 65535) from noise model fitting and from PSNR/SSIM computation.

**Background:**
- Verified via rasterio inspection that several scenes contain pixels at the uint16 maximum value (65535)
- Saturation observed in B04 and B11 for Brest, Toulon, and Panama
- Suez 1 shows no saturation (max ~40k); the desert/canal scene has more uniform reflectance
- Sentinel-2 native quantization is 12-bit (max 4095) scaled to 16-bit container; values at 65535 indicate scaling-stage saturation or particularly bright reflectors (clouds, sun glint, bright urban surfaces)

**Rationale:**
- Saturated pixels distort the Poisson-Gaussian noise variance estimate near the high-value tail (ceiling effects rather than true sensor behavior)
- Including them in PSNR/SSIM would conflate denoising performance with saturation behavior
- Masking is a standard practice in remote sensing image quality work

**Implementation:**
- Apply `pixels_to_use = (raw_band < 65535)` to all metric computations
- Document the saturated-pixel fraction per scene in the experimental results

**Status:** Confirmed (May 24, 2026)

---

## D8 — Acquisition Date Homogeneity

**Date surfaced:** May 24, 2026
**Decision:** Acknowledge as a known limitation; document in Section 1.8.

**Background:**
- All 16 S2-SHIPS scenes were acquired in April 2021
- Six of the 16 share a single acquisition date (2021-04-28, the Suez sequence)
- This means seasonal variation, sun angle variation, and weather variation across the dataset is minimal

**Rationale for acknowledging rather than expanding:**
- Acquiring additional dates would require either (a) extending the S2-SHIPS benchmark with additional acquisitions of the same locations, or (b) using a different benchmark — both of which were considered and not selected
- The praxis claim is about denoising-method comparison, not seasonal robustness
- The same April-2021 imagery applies uniformly across all five denoising methods evaluated, so the comparison among methods is not biased by the date homogeneity

**Implications:**
- Section 1.8 must acknowledge: "The S2-SHIPS benchmark imagery is concentrated in April 2021, with limited seasonal variation. Generalization of method-specific findings to imagery acquired under different seasonal conditions is a matter of empirical extension rather than direct experimental verification within the praxis."

**Status:** Confirmed (May 24, 2026)

---

## Open Questions Summary

The following questions remain open and require praxis advisor input before final methodology lock:

1. **D2:** L1C vs L2A product level treatment
2. **D3:** Final confirmation of ship + moored ship category combination (tentatively decided but should be confirmed)
3. **D4:** Statistical treatment of six dependent Suez scenes
4. **D5:** Native 20m vs S2-SHIPS resampled 10m for B05 and B11

These will be raised in the next advisor meeting and resolved before experimental code freeze.
