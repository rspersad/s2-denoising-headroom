# Data card

**Title.** Sentinel-2 vessel-detection tiles with calibrated Poisson-Gaussian degradation, denoiser outputs and matched detection scores.

**Purpose.** To measure whether denoising selected on perceptual fidelity changes vessel-detection performance (mAP@0.5) on Sentinel-2 imagery, under controlled degradation.

**Source imagery.** Copernicus Sentinel-2 Level-2A, bands B04 (664.6 nm, 10 m), B05 (704.1 nm, 20 m resampled to 10 m), B11 (1613.7 nm, 20 m resampled to 10 m). Scenes: [list the scenes, acquisition dates and tile IDs]. Attribution: Contains modified Copernicus Sentinel data [years].

**Annotations.** Vessel bounding boxes derived from S2-SHIPS (Ciocarlan & Stoian, 2021). Median vessel 4 × 4 pixels at 10 m. Redistribution terms: docs/PERMISSIONS.md.

**Processing.** 640 × 640 tiles at native resolution, stride 512 (128-pixel overlap), edge tiles clamped. Single band replicated to three channels for the detector. Contrast stretch P1.0/P99.5 from water pixels; CLAHE clip 2.5, grid 8.

**Degradation.** Poisson-Gaussian noise, per-band gain α solved from the Sentinel-2 SNR specification at the reference radiance; read noise σr = 5 / 5 / 2 DN (B04 / B05 / B11) by stated assumption (≈ 10 % of total noise variance at the reference radiance). Three severities, three independent draws. Note: gains for B04 and B05 were solved at reference pixel values corresponding to 122 and 105 W m⁻² sr⁻¹ μm⁻¹; the specification values are 108 and 74.6. Effective nominal SNR ≈ 133 (B04) and 97 (B05). Recorded in the Praxis, Table 3.1 note and §5.6.

**Design.** 5 denoisers × 3 bands × 3 severities × 4 held-out scenes × 3 draws = 648 factorial evaluations; 12 clean-reference; 144 severity-matched arm; 804 in total. Nine YOLOv9 detectors (m / c / e × 3 seeds). Measurement resolution 0.005 mAP@0.5.

**Files.** See `docs/VARIABLES.md` for the column dictionary of `results/primary_table.csv` and companions.

**Known limitations.** Synthetic degradation; four held-out scenes; targets at the sensor's resolving limit; reference-radiance discrepancy above.

**Licenses.** Code AGPL-3.0; tables, configs and documentation CC BY 4.0; detector weights AGPL-3.0; imagery under the Copernicus licence; annotation-derived files per docs/PERMISSIONS.md.
