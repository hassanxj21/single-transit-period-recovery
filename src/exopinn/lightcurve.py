"""Transit light-curve forward model and the measurement estimators.

The same two functions are used on synthetic *and* real data:

  * ``extract_window``  - resample a fixed-length window in **absolute days**
  * ``measure_transit`` - depth and half-depth duration from that window

Keeping train-time and inference-time feature extraction identical is not a
nicety. The v3 CNN-PINN generated each synthetic light curve on a grid spanning
+/-1.5 x its own duration, which makes every training example look identical up
to depth: the array carried no absolute timescale, so the CNN could not encode
duration at all. Real light curves were then interpolated over an arbitrary
window in days. That distribution mismatch, not the network, produced the 486%
error on TIC 393818343.
"""

from __future__ import annotations

import numpy as np

from .constants import R_SUN_AU
from .physics import eccentricity_duration_factor, semi_major_axis

# Canonical extraction geometry. Every light curve fed to the network -
# synthetic or real - is this window, centred on the transit.
WINDOW_DAYS = 2.5
N_POINTS = 128


# --------------------------------------------------------------------------
# Forward model
# --------------------------------------------------------------------------
def occulted_fraction(z, k):
    """Fraction of a uniform stellar disk hidden by an opaque planet.

    z is sky-projected separation in units of R*, k = Rp/R*. Standard
    circle-circle overlap (Mandel & Agol 2002, uniform-source case).
    """
    z = np.asarray(z, float)
    k = np.asarray(k, float)
    out = np.zeros(np.broadcast(z, k).shape, float)

    full = z <= np.abs(1.0 - k)
    out = np.where(full, np.minimum(k, 1.0) ** 2, out)

    partial = (z > np.abs(1.0 - k)) & (z < 1.0 + k)
    if np.any(partial):
        zp = np.where(partial, z, 1.0)
        kp = np.where(partial, k, 0.5)
        kap0 = np.arccos(np.clip((kp**2 + zp**2 - 1.0) / (2.0 * kp * zp), -1.0, 1.0))
        kap1 = np.arccos(np.clip((1.0 - kp**2 + zp**2) / (2.0 * zp), -1.0, 1.0))
        tri = np.sqrt(np.clip(4.0 * zp**2 - (1.0 + zp**2 - kp**2) ** 2, 0.0, None)) / 2.0
        out = np.where(partial, (kp**2 * kap0 + kap1 - tri) / np.pi, out)

    return np.where(z <= k - 1.0, 1.0, out)


def simulate_transit(t, t0, period, stellar_mass, stellar_radius, k, b, ecc=0.0, omega=np.pi / 2, u1=0.4, u2=0.25):
    """Normalised flux on time grid ``t`` (days) for one transit.

    Eccentricity enters through the sky-crossing speed at conjunction, which is
    the only way it can affect a single transit: the projected separation is
    x(t) = (a/R*) sin(2 pi f_e (t - t0) / P) with f_e = 1 / duration factor.
    Limb darkening uses the quadratic law under the small-planet approximation,
    which is what gives the ingress/egress its curvature - the shape cue the
    CNN branch exists to read.
    """
    a_over_r = semi_major_axis(period, stellar_mass) / (stellar_radius * R_SUN_AU)
    f_e = 1.0 / eccentricity_duration_factor(ecc, omega)

    x = a_over_r * np.sin(2.0 * np.pi * f_e * (np.asarray(t, float) - t0) / period)
    z = np.sqrt(x**2 + b**2)
    # Only the near side of the orbit transits.
    z = np.where(np.cos(2.0 * np.pi * f_e * (np.asarray(t, float) - t0) / period) > 0, z, 10.0)

    mu = np.sqrt(np.clip(1.0 - np.minimum(z, 1.0) ** 2, 0.0, 1.0))
    intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    mean_intensity = 1.0 - u1 / 3.0 - u2 / 6.0

    return 1.0 - occulted_fraction(z, k) * intensity / mean_intensity


# --------------------------------------------------------------------------
# Measurement (identical for synthetic and real light curves)
# --------------------------------------------------------------------------
def _boxcar(x, w):
    if w < 2:
        return x
    kern = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(x, pad, mode="edge"), kern, mode="same")[pad : pad + len(x)]


def measure_transit(t, flux, smooth=5):
    """Return (depth, fwhm_duration_hours, t_center) from a windowed light curve.

    Duration is the full width at half depth, measured by linear interpolation
    of the half-depth crossings on either side of the minimum. This is what a
    threshold measurement on real data gives, and it is the quantity
    ``physics.transit_duration(..., contact="fwhm")`` models.

    Returns nan duration if no clean pair of crossings exists.
    """
    t = np.asarray(t, float)
    f = _boxcar(np.asarray(flux, float), smooth)

    baseline = np.median(f)
    imin = int(np.argmin(f))
    depth = float(baseline - f[imin])
    if depth <= 0:
        return 0.0, np.nan, float(t[imin])

    half = baseline - depth / 2.0

    def crossing(idx_range, direction):
        prev = imin
        for i in idx_range:
            if f[i] >= half:
                # linear interpolation between i and prev
                f0, f1 = f[i], f[prev]
                if f0 == f1:
                    return t[i]
                return t[i] + (half - f0) * (t[prev] - t[i]) / (f1 - f0)
            prev = i
        return np.nan

    t_left = crossing(range(imin - 1, -1, -1), -1)
    t_right = crossing(range(imin + 1, len(f)), +1)

    if np.isnan(t_left) or np.isnan(t_right):
        return depth, np.nan, float(t[imin])

    return depth, float((t_right - t_left) * 24.0), float(0.5 * (t_left + t_right))


def extract_window(t, flux, t0, window_days=WINDOW_DAYS, n_points=N_POINTS):
    """Resample ``flux`` onto ``n_points`` uniformly spaced samples covering
    ``[t0 - window/2, t0 + window/2]`` days. Gaps are filled with the local
    median so the array length is always fixed.

    Returns (t_grid, flux_grid) or (None, None) if the window is mostly empty.
    """
    t = np.asarray(t, float)
    flux = np.asarray(flux, float)
    grid = np.linspace(t0 - window_days / 2.0, t0 + window_days / 2.0, n_points)

    inside = (t >= grid[0]) & (t <= grid[-1])
    if inside.sum() < n_points // 4:
        return None, None

    resampled = np.interp(grid, t[inside], flux[inside], left=np.nan, right=np.nan)
    fill = np.nanmedian(resampled)
    return grid, np.nan_to_num(resampled, nan=fill)
