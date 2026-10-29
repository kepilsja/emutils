"""Apodization profiles of grating couplers: per-segment pitches and fill factors."""
import numpy as np


def phase_matched_pitch(tooth_ff, wavelength, n_eff_tooth, n_eff_trench, theta, n_clad=1.0):
    """
    Pitch satisfying the phase matching (Bragg) condition of a two-section grating,

        pitch = wavelength / (n_eff - n_clad*sin(theta)),
        n_eff = F*n_eff_tooth + (1 - F)*n_eff_trench,

    where F is `tooth_ff`, the unetched fraction of the period (scalar `fill_factor` of
    `GratingGeometry`), n_eff_tooth / n_eff_trench are effective indices of the unetched
    and etched slab (e.g. from `emutils.eim.solve_1D_analytic`) and theta (deg) is the fiber
    angle in the cladding (n_clad*sin(theta) is conserved, so the angle in air with n_clad=1
    can be used as well). Works element-wise on arrays of `tooth_ff`.
    """
    tooth_ff = np.asarray(tooth_ff, dtype=float)
    n_eff = tooth_ff * n_eff_tooth + (1 - tooth_ff) * n_eff_trench
    denominator = n_eff - n_clad * np.sin(np.radians(theta))
    if np.any(denominator <= 0):
        raise ValueError("No phase matched pitch: n_eff must exceed n_clad*sin(theta).")
    return wavelength / denominator


def linear_apodization(n_segments, ff_start, rate, wavelength, n_eff_tooth, n_eff_trench, theta,
                       n_clad=1.0, ff_min=0.0, min_feature=0.0):
    """
    Linearly apodized two-section grating with phase matched pitch of every segment
    (Ding et al., Opt. Lett. 39, 5348 (2014)):

        F_i = ff_start - rate*x_i,    pitch_i = phase_matched_pitch(F_i, ...),

    where x_i is the start of i-th segment measured from the grating start and F is the
    unetched (tooth) fraction of the period. F_i is clipped to [ff_min, 1] and, with
    `min_feature` > 0, so that both the trench and the tooth of every segment are at least
    `min_feature` wide. See `phase_matched_pitch` for the remaining arguments.

    Args:
        n_segments (int): Number of grating segments.
        ff_start (float): Tooth fraction at the grating start (weak scattering for F close to 1).
        rate (float): Decrease of the tooth fraction per unit length in 1/m, e.g. 0.025/um.

    Returns:
        (pitches, fill_factors): arrays of shape (n_segments,) and (n_segments, 2) with rows
        [trench, tooth], to be used as `pitch` and `fill_factor` of `GratingCoupler2D` or
        `geom.update(n_segments=..., pitch=..., fill_factor=...)`.
    """
    s = n_clad * np.sin(np.radians(theta))
    dn = n_eff_tooth - n_eff_trench
    # tooth width F*pitch grows and trench width (1 - F)*pitch shrinks monotonically with F,
    # bounds below are solutions of these widths equal to min_feature
    ff_low = max(ff_min, min_feature * (n_eff_trench - s) / (wavelength - min_feature * dn))
    ff_high = min(1.0, (wavelength - min_feature * (n_eff_trench - s)) / (wavelength + min_feature * dn))
    if ff_low > ff_high:
        raise ValueError(
            f"No tooth fraction satisfies ff_min={ff_min} and min_feature={min_feature}, "
            f"allowed range is [{ff_low}, {ff_high}]."
        )

    pitches = np.empty(n_segments)
    tooth = np.empty(n_segments)
    x = 0.0
    for i in range(n_segments):
        tooth[i] = np.clip(ff_start - rate * x, ff_low, ff_high)
        pitches[i] = phase_matched_pitch(tooth[i], wavelength, n_eff_tooth, n_eff_trench, theta, n_clad)
        x += pitches[i]

    return pitches, np.column_stack((1 - tooth, tooth))
