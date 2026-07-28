"""Orbital and transit physics.

Everything the project needs to go from (transit observables + stellar properties)
to an orbital period, and back. Both NumPy and PyTorch implementations are provided
so the same equations drive synthetic data generation and the PINN physics loss.

Sign conventions
----------------
`omega` is the argument of periastron in radians, defined so that omega = +pi/2
places the transit at periastron (fastest sky motion, shortest duration) and
omega = -pi/2 places it at apoastron (slowest, longest duration).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .constants import FOURPI2, G, R_SUN_AU

__all__ = [
    "semi_major_axis",
    "period_from_semi_major_axis",
    "eccentricity_duration_factor",
    "transit_duration",
    "period_from_duration",
    "min_physical_period",
    "is_physically_possible",
    "equilibrium_temperature",
    "transit_duration_torch",
    "semi_major_axis_torch",
]


# --------------------------------------------------------------------------
# Kepler's Third Law
# --------------------------------------------------------------------------
def semi_major_axis(period_days, stellar_mass_msun):
    """a [AU] from Kepler's Third Law: a = (G M P^2 / 4pi^2)^(1/3)."""
    return (G * np.asarray(stellar_mass_msun, float) * np.asarray(period_days, float) ** 2 / FOURPI2) ** (1.0 / 3.0)


def period_from_semi_major_axis(a_au, stellar_mass_msun):
    """P [days] from a [AU]."""
    return np.sqrt(FOURPI2 * np.asarray(a_au, float) ** 3 / (G * np.asarray(stellar_mass_msun, float)))


# --------------------------------------------------------------------------
# Transit duration geometry
# --------------------------------------------------------------------------
def eccentricity_duration_factor(ecc=0.0, omega=np.pi / 2):
    """Multiplicative correction to transit duration for an eccentric orbit.

    T_ecc / T_circ = sqrt(1 - e^2) / (1 + e sin(omega))

    Ranges from (1-e)/sqrt(1-e^2) at periastron to (1+e)/sqrt(1-e^2) at
    apoastron, so a modest e = 0.3 spans 0.73x to 1.36x. This term is the
    single largest source of period error in single-transit recovery,
    because period scales as duration cubed.
    """
    ecc = np.asarray(ecc, float)
    return np.sqrt(1.0 - ecc**2) / (1.0 + ecc * np.sin(omega))


def transit_duration(
    period_days,
    stellar_mass_msun,
    stellar_radius_rsun,
    k=0.0,
    b=0.0,
    ecc=0.0,
    omega=np.pi / 2,
    hours=True,
    contact="14",
):
    """Transit duration between contact points.

    T14 = (P / pi) * arcsin[ (R*/a) * sqrt((1+k)^2 - b^2) / sin(i) ] * f(e, omega)

    We drop the 1/sin(i) factor, which is <1% for the a/R* regime of interest.

    Parameters
    ----------
    k : float or array
        Planet-to-star radius ratio Rp/R*. Recoverable from transit depth as
        k = sqrt(depth). Omitting this term (i.e. using sqrt(1 - b^2)) biases
        durations low by ~k, which is 21% for a depth of 4.45%.
    b : float or array
        Impact parameter in units of R*. A grazing transit (b -> 1+k) shortens
        the duration; b > 1 + k means no transit at all.
    contact : {"14", "23", "fwhm"}
        "14" is first-to-fourth contact (total duration), "23" is the flat
        bottom, "fwhm" is the full width at half depth = (T14 + T23) / 2.
        "fwhm" is what a simple half-depth threshold measures on a real light
        curve, so it is the quantity the pipeline actually compares against.

    Returns nan where the geometry admits no transit.
    """
    period_days = np.asarray(period_days, float)
    a = semi_major_axis(period_days, stellar_mass_msun)
    r_star_au = np.asarray(stellar_radius_rsun, float) * R_SUN_AU

    if contact == "fwhm":
        kw = dict(ecc=ecc, omega=omega, hours=hours)
        t14 = transit_duration(period_days, stellar_mass_msun, stellar_radius_rsun, k, b, contact="14", **kw)
        t23 = transit_duration(period_days, stellar_mass_msun, stellar_radius_rsun, k, b, contact="23", **kw)
        return 0.5 * (t14 + np.nan_to_num(t23, nan=0.0))

    sign = 1.0 if contact == "14" else -1.0
    chord_sq = (1.0 + sign * np.asarray(k, float)) ** 2 - np.asarray(b, float) ** 2
    chord = np.sqrt(np.where(chord_sq > 0, chord_sq, np.nan))

    arg = (r_star_au / a) * chord
    arg = np.where(np.abs(arg) < 1.0, arg, np.nan)  # planet inside the star -> unphysical

    dur_days = (period_days / np.pi) * np.arcsin(arg) * eccentricity_duration_factor(ecc, omega)
    return dur_days * 24.0 if hours else dur_days


def period_from_duration(
    duration_hours,
    stellar_mass_msun,
    stellar_radius_rsun,
    k=0.0,
    b=0.0,
    ecc=0.0,
    omega=np.pi / 2,
    p_lo=0.05,
    p_hi=1.0e5,
    contact="14",
):
    """Invert the duration relation for the orbital period (analytic baseline).

    This is the closed-form single-transit period estimator. Because
    T14 ~ P^(1/3) at fixed stellar density, the inverse scales as

        P  ~  (pi G M* / 4) * T14^3 / (R*^3 * ((1+k)^2 - b^2)^(3/2))

    i.e. **period error is the cube of duration error**. A 10% duration error
    is a 33% period error; a 60% error is a factor of 4. This function is the
    physics baseline the CNN-PINN must beat, and the reason a point estimate
    from a single transit is not a defensible product.

    Scalar-only (uses a 1-D root find); vectorise with np.vectorize if needed.
    """
    def residual(p):
        d = transit_duration(p, stellar_mass_msun, stellar_radius_rsun, k, b, ecc, omega, contact=contact)
        return (0.0 if np.isnan(d) else float(d)) - duration_hours

    if residual(p_lo) > 0:
        return np.nan  # duration too short for any orbit around this star
    try:
        return float(brentq(residual, p_lo, p_hi, xtol=1e-8, rtol=1e-10))
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------
# Physical plausibility of a candidate period
# --------------------------------------------------------------------------
def min_physical_period(stellar_mass_msun, stellar_radius_rsun, roche_factor=1.0):
    """Shortest period whose orbit still clears the stellar surface (a = R*).

    Any reported period below this implies the planet orbits inside its star.
    Set roche_factor ~ 2.0 for a tidal-disruption floor rather than a bare
    surface-grazing floor.
    """
    a_min = roche_factor * np.asarray(stellar_radius_rsun, float) * R_SUN_AU
    return period_from_semi_major_axis(a_min, stellar_mass_msun)


def is_physically_possible(period_days, stellar_mass_msun, stellar_radius_rsun, roche_factor=1.0):
    """True if the implied orbit lies outside the star. Returns (ok, a_over_rstar)."""
    a = semi_major_axis(period_days, stellar_mass_msun)
    a_over_r = a / (np.asarray(stellar_radius_rsun, float) * R_SUN_AU)
    return a_over_r > roche_factor, a_over_r


# --------------------------------------------------------------------------
# Habitability
# --------------------------------------------------------------------------
def equilibrium_temperature(teff_k, stellar_radius_rsun, period_days, stellar_mass_msun, albedo=0.3):
    """T_eq = T* sqrt(R* / 2a) (1 - A)^(1/4), with a from Kepler's Third Law."""
    a = semi_major_axis(period_days, stellar_mass_msun)
    r_star_au = np.asarray(stellar_radius_rsun, float) * R_SUN_AU
    return np.asarray(teff_k, float) * np.sqrt(r_star_au / (2.0 * a)) * (1.0 - albedo) ** 0.25


# --------------------------------------------------------------------------
# Torch mirrors (used inside the PINN physics loss; must stay differentiable)
# --------------------------------------------------------------------------
def semi_major_axis_torch(period_days, stellar_mass_msun):
    import torch  # local import so NumPy-only scripts don't pay for it

    return (G * stellar_mass_msun * period_days**2 / FOURPI2) ** (1.0 / 3.0)


def transit_duration_torch(period_days, stellar_mass_msun, stellar_radius_rsun, k, b, hours=True, contact="fwhm"):
    """Differentiable circular-orbit duration in hours.

    Eccentricity is deliberately excluded: e and omega are unobservable from a
    single transit, so the physics loss anchors to the circular case and the
    model's predicted variance absorbs the spread.
    """
    import torch

    a = semi_major_axis_torch(period_days, stellar_mass_msun)
    r_star_au = stellar_radius_rsun * R_SUN_AU

    def _one(sign):
        chord = torch.sqrt(torch.clamp((1.0 + sign * k) ** 2 - b**2, min=1e-8))
        arg = torch.clamp((r_star_au / a) * chord, -0.999999, 0.999999)
        return (period_days / np.pi) * torch.arcsin(arg)

    if contact == "fwhm":
        dur = 0.5 * (_one(1.0) + _one(-1.0))
    else:
        dur = _one(1.0 if contact == "14" else -1.0)
    return dur * 24.0 if hours else dur
