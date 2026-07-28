"""Box Least Squares baseline, including the search-range dependency experiment."""

from __future__ import annotations

import numpy as np
from astropy.timeseries import BoxLeastSquares


def run_bls(time, flux, p_min, p_max, n_periods=20000, durations=(0.05, 0.1, 0.2, 0.3, 0.5)):
    """Return the best-fit period and diagnostics over one search grid.

    ``edge_fraction`` reports how close the answer sits to a grid boundary:
    a value near 0 or 1 means BLS returned the edge of the range it was handed,
    which is the signature of it having found nothing.
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    ok = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[ok], flux[ok]

    grid = np.linspace(p_min, p_max, n_periods)

    # astropy requires every trial duration to be shorter than the shortest
    # trial period. Keep only the durations that are physically sensible for
    # this grid, and fall back to a short one if the filter empties the list.
    dur = np.asarray([d for d in durations if d < 0.5 * p_min], float)
    if dur.size == 0:
        dur = np.array([0.1 * p_min])

    bls = BoxLeastSquares(time, flux)
    res = bls.power(grid, dur)

    power = np.asarray(res.power, float)
    i = int(np.nanargmax(power))
    best = float(res.period[i])

    med, mad = np.nanmedian(power), 1.4826 * np.nanmedian(np.abs(power - np.nanmedian(power)))
    return {
        "period": best,
        "power": float(power[i]),
        "sde": float((power[i] - med) / mad) if mad > 0 else np.nan,
        "depth": float(res.depth[i]),
        "duration": float(res.duration[i]),
        "t0": float(res.transit_time[i]),
        "range": (p_min, p_max),
        "edge_fraction": float((best - p_min) / (p_max - p_min)),
        "at_edge": bool(min(best - p_min, p_max - best) < 3 * (p_max - p_min) / n_periods),
        "grid": grid,
        "power_spectrum": power,
        "n_transits_in_baseline": float((time.max() - time.min()) / best),
    }


def range_dependency(time, flux, ranges, **kwargs):
    """Run BLS over several search ranges to expose its dependence on the prior.

    The point of the experiment: if the reported period tracks the search range
    rather than the data, BLS is not measuring anything.
    """
    return [run_bls(time, flux, lo, hi, **kwargs) for lo, hi in ranges]
