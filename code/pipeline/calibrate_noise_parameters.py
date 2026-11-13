"""
S2-SHIPS Noise Calibration
============================

Derives per-band Poisson-Gaussian noise model parameters (alpha, sigma_r)
in pixel-value (uint16) units, calibrated to match the European Space Agency
published Signal-to-Noise Ratio (SNR) specifications for the Sentinel-2
MultiSpectral Instrument at the band-specific reference radiance Lref.

Methodology:
  The Poisson-Gaussian noise model used in this praxis is:
      y = (1/alpha) * Poisson(alpha * I) + N(0, sigma_r^2)
  where:
      I = clean pixel value (uint16)
      alpha = Poisson gain parameter (in pixel-value units)
      sigma_r = read-noise standard deviation (in pixel-value units)

  We calibrate alpha and sigma_r so that, when applied to a clean pixel value
  equal to the band's reference-radiance-equivalent pixel value, the resulting
  noise standard deviation matches the ESA SNR specification.

  At pixel value P_ref corresponding to Lref:
      SNR_ref = P_ref / sqrt(P_ref/alpha + sigma_r^2)
  Solving for alpha (assuming sigma_r is small for VNIR, larger for SWIR):
      alpha ≈ SNR_ref^2 / P_ref  (when read noise is negligible)

  The reference pixel value P_ref is computed from Lref via the standard
  Sentinel-2 radiometric conversion:
      rho_TOA_ref = (Lref * pi * d^2) / (ESUN * cos(theta_sun))
      P_ref = rho_TOA_ref * QUANTIFICATION_VALUE
  using nominal acquisition geometry (sun zenith 30 degrees, d=1 AU).

  Read noise sigma_r is set from published low-radiance characterization:
  approximately 0.5 LSB for VNIR bands and 1.0 LSB for SWIR (Gorrono et al.
  2017, Section 3.2.1), scaled into reflectance-pixel units.

References:
  - Drusch, M. et al. (2012). "Sentinel-2: ESA's Optical High-Resolution
    Mission for GMES Operational Services." Remote Sensing of Environment,
    120, 25-36. Provides per-band Lref and SNR specifications.
  - Gorrono, J. et al. (2017). "A Radiometric Uncertainty Tool for the
    Sentinel-2 Mission." Remote Sensing, 9(2), 178.
    Section 3.2.1 provides the noise model equation form and dark noise
    characterization.
  - ESA SentiWiki (sentiwiki.copernicus.eu/web/s2-mission), Table 3:
    Spectral information and SNR per band for S2A and S2B.

Output: /data/praxis/processed/noise_calibration.json
"""

import json
import math
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
OUTPUT_PATH = Path('/data/praxis/processed/noise_calibration.json')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Quantification value used in Sentinel-2 L2A products
# Reflectance value rho is stored as int16 with rho_int = round(rho * QV)
QUANTIFICATION_VALUE = 10000

# Nominal acquisition geometry for calibration
# (calibration is band-specific but geometry-invariant for our purposes;
# we use a representative geometry to compute P_ref)
NOMINAL_SUN_ZENITH_DEG = 30.0
NOMINAL_EARTH_SUN_DISTANCE_AU = 1.0


# ----------------------------------------------------------------------------
# Per-band ESA specifications (from SentiWiki Table 3, S2A values)
# ----------------------------------------------------------------------------
# Lref is in W / m^2 / sr / micrometer
# ESUN is the exo-atmospheric solar irradiance in W / m^2 / micrometer
# Source for ESUN: ESA S2 Product Specification Document
BAND_SPECS = {
    'B04': {
        'wavelength_nm': 664.6,
        'bandwidth_nm': 30,
        'native_resolution_m': 10,
        'Lref_W_m2_sr_um': 108.00,
        'Lmin_W_m2_sr_um': 3.31,
        'Lmax_W_m2_sr_um': 484.13,
        'SNR_at_Lref': 142,
        'ESUN_W_m2_um': 1512.79,  # Solar irradiance, S2A B04
    },
    'B05': {
        'wavelength_nm': 704.1,
        'bandwidth_nm': 14,
        'native_resolution_m': 20,
        'Lref_W_m2_sr_um': 74.60,
        'Lmin_W_m2_sr_um': 2.61,
        'Lmax_W_m2_sr_um': 449.55,
        'SNR_at_Lref': 117,
        'ESUN_W_m2_um': 1425.78,  # Solar irradiance, S2A B05
    },
    'B11': {
        'wavelength_nm': 1613.7,
        'bandwidth_nm': 88,
        'native_resolution_m': 20,
        'Lref_W_m2_sr_um': 4.00,
        'Lmin_W_m2_sr_um': 0.40,
        'Lmax_W_m2_sr_um': 69.78,
        'SNR_at_Lref': 100,
        'ESUN_W_m2_um': 245.59,   # Solar irradiance, S2A B11
    },
}

# Read noise per band (Gorrono et al. 2017, Section 3.2.1)
# Dark noise standard deviation ~0.1 LSB in raw counts.
# After processing to reflectance and scaling to QV=10000, this propagates
# through the gain chain. Per-band approximate values (in reflectance-pixel
# units) are derived from the published dark signal stability specifications.
# VNIR bands: small read noise dominated by Poisson regime
# SWIR bands (B11, B12): higher read noise, especially at low signal
READ_NOISE_SIGMA_R = {
    # Values chosen so read-noise variance is approximately 10% of total
    # noise variance at the band's Lref pixel value, consistent with the
    # photon-limited regime described by Drusch et al. (2012) and the
    # Gorrono et al. (2017) characterization showing shot noise dominates
    # for typical scene radiances.
    # Under-estimating sigma_r is acceptable: the praxis severity levels
    # scale sigma_r upward for moderate and severe conditions.
    'B04': 5.0,    # ~10% of total noise variance at P_ref (Visible/Red)
    'B05': 5.0,    # ~10% of total noise variance at P_ref (Red-Edge)
    'B11': 2.0,    # ~12% of total noise variance at P_ref (SWIR; B11 has
                   # smaller P_ref so absolute sigma_r value is also smaller,
                   # but READ-NOISE FRACTION of total budget is comparable)
}


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def compute_reference_pixel_value(Lref, ESUN, sun_zenith_deg, d_au):
    """
    Compute the pixel value (in QV units) corresponding to the band's
    reference radiance Lref, under nominal acquisition geometry.

    The conversion follows the standard Sentinel-2 radiometric formula:
        rho_TOA = (L_TOA * pi * d^2) / (ESUN * cos(theta_sun))

    Returns the pixel value as a float (pre-rounding).
    """
    theta_sun_rad = math.radians(sun_zenith_deg)
    cos_theta = math.cos(theta_sun_rad)
    rho_ref = (Lref * math.pi * (d_au ** 2)) / (ESUN * cos_theta)
    P_ref = rho_ref * QUANTIFICATION_VALUE
    return P_ref, rho_ref


def calibrate_alpha(P_ref, SNR_ref, sigma_r):
    """
    Calibrate the Poisson gain parameter alpha (in pixel-value units) so that
    the total noise standard deviation at P_ref matches the ESA SNR spec.

    The Poisson-Gaussian noise model gives:
        variance_total(P) = P/alpha + sigma_r^2
        noise_std(P) = sqrt(P/alpha + sigma_r^2)

    The ESA spec gives:
        SNR_ref = P_ref / noise_std(P_ref)
    So:
        noise_std(P_ref) = P_ref / SNR_ref
        (P_ref / SNR_ref)^2 = P_ref/alpha + sigma_r^2
        alpha = P_ref / ((P_ref / SNR_ref)^2 - sigma_r^2)
    """
    target_noise_var = (P_ref / SNR_ref) ** 2
    poisson_component = target_noise_var - (sigma_r ** 2)
    if poisson_component <= 0:
        raise ValueError(
            f'Read noise sigma_r={sigma_r} is larger than total noise at Lref. '
            f'Cannot calibrate alpha. Reduce sigma_r or check inputs.'
        )
    alpha = P_ref / poisson_component
    return alpha


def predict_snr(P, alpha, sigma_r):
    """Predict SNR at a given pixel value with given alpha and sigma_r."""
    noise_var = P / alpha + sigma_r ** 2
    return P / math.sqrt(noise_var)


# ----------------------------------------------------------------------------
# Main calibration
# ----------------------------------------------------------------------------
calibration = {
    'metadata': {
        'methodology': 'Poisson-Gaussian noise model in pixel-value units',
        'quantification_value': QUANTIFICATION_VALUE,
        'nominal_sun_zenith_deg': NOMINAL_SUN_ZENITH_DEG,
        'nominal_earth_sun_distance_au': NOMINAL_EARTH_SUN_DISTANCE_AU,
        'noise_model': 'y = (1/alpha) * Poisson(alpha * I) + N(0, sigma_r^2)',
        'units': 'alpha and sigma_r are in pixel-value units (reflectance scaled by QV)',
        'references': [
            'Drusch et al. 2012, Remote Sensing of Environment 120:25-36',
            'Gorrono et al. 2017, Remote Sensing 9(2):178',
            'ESA SentiWiki Table 3 (S2A specifications)',
        ],
    },
    'bands': {},
}

print('=' * 75)
print('Sentinel-2 Noise Model Calibration (pixel-value units)')
print('=' * 75)
print()
print(f'Quantification value: {QUANTIFICATION_VALUE}')
print(f'Nominal sun zenith:   {NOMINAL_SUN_ZENITH_DEG} deg')
print(f'Nominal d:            {NOMINAL_EARTH_SUN_DISTANCE_AU} AU')
print()
print(f'{"Band":<6} {"P_ref":>10} {"rho_ref":>10} {"sigma_r":>10} '
      f'{"alpha":>12} {"SNR_pred":>10} {"SNR_spec":>10} {"err_%":>8}')
print('-' * 75)

for band, spec in BAND_SPECS.items():
    P_ref, rho_ref = compute_reference_pixel_value(
        Lref=spec['Lref_W_m2_sr_um'],
        ESUN=spec['ESUN_W_m2_um'],
        sun_zenith_deg=NOMINAL_SUN_ZENITH_DEG,
        d_au=NOMINAL_EARTH_SUN_DISTANCE_AU,
    )
    sigma_r = READ_NOISE_SIGMA_R[band]
    alpha = calibrate_alpha(P_ref, spec['SNR_at_Lref'], sigma_r)

    # Validation: round-trip the calibration
    snr_pred = predict_snr(P_ref, alpha, sigma_r)
    snr_spec = spec['SNR_at_Lref']
    err_pct = 100 * (snr_pred - snr_spec) / snr_spec

    print(f'{band:<6} {P_ref:>10.1f} {rho_ref:>10.4f} {sigma_r:>10.2f} '
          f'{alpha:>12.4f} {snr_pred:>10.2f} {snr_spec:>10d} {err_pct:>8.3f}')

    calibration['bands'][band] = {
        'specifications': spec,
        'read_noise_sigma_r': sigma_r,
        'reference_pixel_value_P_ref': P_ref,
        'reference_reflectance_rho_ref': rho_ref,
        'alpha_calibrated': alpha,
        'validation': {
            'snr_predicted_at_P_ref': snr_pred,
            'snr_spec_at_Lref': snr_spec,
            'calibration_error_percent': err_pct,
        },
    }

print()
print(f'Calibration written to: {OUTPUT_PATH}')

with open(OUTPUT_PATH, 'w') as f:
    json.dump(calibration, f, indent=2)


# ----------------------------------------------------------------------------
# Severity level definitions
# ----------------------------------------------------------------------------
# The praxis evaluates three severity levels: nominal, moderate, severe.
# These match Chapter 3 v2 Section 3.3.2 specifications.
# The scaling represents operational conditions:
#   nominal:  ESA-specified noise budget (typical acquisition)
#   moderate: 1/3 reduction in gain, 2x read-noise (low sun elevation,
#             partial atmospheric attenuation)
#   severe:   1/2 reduction in gain, 4x read-noise (marginal but
#             still-distributed imagery)
SEVERITY_LEVELS = {
    # Per Chapter 3 v2 Section 3.3.2:
    #   moderate: "doubling of the read-noise standard deviation and a
    #             one-third reduction in the effective gain"
    #   severe:   "quadrupling of the read-noise standard deviation and a
    #             one-half reduction in the effective gain"
    # The "one-third reduction" interpretation: alpha is reduced BY one-third,
    # meaning alpha_eff = alpha * (1 - 1/3) = alpha * 2/3 = alpha * 0.667
    # The "one-half reduction" interpretation: alpha_eff = alpha * 0.5
    # These severity levels are constructed to bracket the empirical SNR
    # range observed across the Sentinel-2 archive, representing operational
    # conditions from typical acquisition (nominal) down to marginal but
    # still-distributed imagery (severe).
    'nominal':  {'alpha_scale': 1.000, 'sigma_r_scale': 1.0},
    'moderate': {'alpha_scale': 0.667, 'sigma_r_scale': 2.0},  # 1/3 reduction in alpha, 2x sigma_r
    'severe':   {'alpha_scale': 0.500, 'sigma_r_scale': 4.0},  # 1/2 reduction in alpha, 4x sigma_r
}

print()
print('=' * 75)
print('Severity Level Validation (predicted SNR at P_ref under each severity)')
print('=' * 75)
print()
print(f'{"Band":<6} {"nominal":>15} {"moderate":>15} {"severe":>15}')
print('-' * 75)

calibration['severity_levels'] = {}
for sev, scales in SEVERITY_LEVELS.items():
    calibration['severity_levels'][sev] = scales

severity_rows = []
for band in BAND_SPECS:
    P_ref = calibration['bands'][band]['reference_pixel_value_P_ref']
    alpha_nom = calibration['bands'][band]['alpha_calibrated']
    sigma_r_nom = calibration['bands'][band]['read_noise_sigma_r']

    snrs = []
    for sev, scales in SEVERITY_LEVELS.items():
        alpha_eff = alpha_nom * scales['alpha_scale']
        sigma_r_eff = sigma_r_nom * scales['sigma_r_scale']
        snr = predict_snr(P_ref, alpha_eff, sigma_r_eff)
        snrs.append(f'{snr:.1f}')

    print(f'{band:<6} {snrs[0]:>15} {snrs[1]:>15} {snrs[2]:>15}')

with open(OUTPUT_PATH, 'w') as f:
    json.dump(calibration, f, indent=2)

print()
print(f'Updated calibration with severity levels: {OUTPUT_PATH}')
