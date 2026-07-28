"""Savitzky-Golay detrending on plain arrays.

Deliberately NOT lightkurve's `flatten()`. Injection-recovery only removes the
domain gap if the injected transit is processed by *exactly* the same code as a
real one, so both paths call this function. Anything lightkurve does that this
does not would reappear as a sim-to-real gap.

Gaps are handled by splitting into contiguous chunks and detrending each
separately, so the filter never interpolates across a downlink gap.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def window_cadences(t, window_days):
    """Odd filter length in cadences for a window specified in days."""
    dt = float(np.median(np.diff(np.sort(np.asarray(t, float)))))
    if not np.isfinite(dt) or dt <= 0:
        return 11
    return max(11, int(round(window_days / dt)) | 1)


def _contiguous_chunks(t, gap_factor=5.0):
    t = np.asarray(t, float)
    if len(t) < 2:
        return [slice(0, len(t))]
    dt = np.diff(t)
    med = np.median(dt)
    breaks = np.flatnonzero(dt > gap_factor * med) + 1
    edges = [0, *breaks.tolist(), len(t)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a > 0]


def savgol_detrend(t, f, window_days=1.0, transit_mask=None, polyorder=2,
                   return_trend=False, niters=3, sigma=3.0):
    """Return flux divided by its smooth trend.

    ``transit_mask`` is True *in transit*. Those cadences are excluded from the
    trend fit - the trend is interpolated across them from the surrounding
    baseline - but they are retained in the output. Without this the filter
    absorbs the transit it is supposed to preserve.

    ``niters`` rounds of sigma-clipping additionally drop points that fall more
    than ``sigma`` below the current trend. This matters more than it sounds:
    omitting it made the measured depth of NGTS-38 b 0.118% against a published
    0.349%, because the first-pass mask is sized from a first-pass duration that
    the unprotected filter had already shrunk.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    mask = np.zeros(len(t), bool) if transit_mask is None else np.asarray(transit_mask, bool)

    trend = np.ones(len(t), float)
    for sl in _contiguous_chunks(t):
        ts, fs, ms = t[sl], f[sl], mask[sl]
        n = len(ts)
        if n < 13:
            trend[sl] = np.median(fs) if n else 1.0
            continue

        wl = min(window_cadences(ts, window_days), (n - 1) | 1)
        if wl < polyorder + 2:
            trend[sl] = np.median(fs)
            continue

        # Replace excluded flux with an interpolation of the baseline before
        # filtering, so dips cannot pull the trend down. Repeat with points that
        # the current trend says are outliers, which catches transits the caller
        # did not mask.
        excluded = ms.copy()
        tr = None
        for _ in range(max(1, niters)):
            fit = fs.copy()
            if excluded.any() and (~excluded).sum() >= 2:
                fit[excluded] = np.interp(ts[excluded], ts[~excluded], fs[~excluded])
            tr = savgol_filter(fit, wl, polyorder, mode="interp")

            resid = fs - tr
            scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
            if not np.isfinite(scale) or scale <= 0:
                break
            new_excluded = ms | (resid < -sigma * scale)
            if new_excluded.sum() == excluded.sum():
                break
            excluded = new_excluded

        trend[sl] = tr

    trend = np.where(np.abs(trend) < 1e-8, 1.0, trend)
    flat = f / trend
    return (flat, trend) if return_trend else flat
