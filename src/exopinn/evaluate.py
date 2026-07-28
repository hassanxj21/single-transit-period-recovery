"""Metrics and the analytic baseline the CNN-PINN has to beat.

The analytic estimator is the honest comparison point, not BLS. BLS cannot run
on a single transit at all; the circular-orbit inversion can, and it is what a
domain scientist would reach for first. If the network does not beat it, the
network is not earning its place.
"""

from __future__ import annotations

import numpy as np

from .physics import period_from_duration


def analytic_periods(scalars, b=0.0, ecc=0.0):
    """Circular, central-transit period estimate for each row of `scalars`.

    scalars columns: [duration_fwhm_hr, depth, mass, radius].
    Depth gives k = sqrt(depth); b and e are unobservable from one transit and
    default to the maximally-agnostic circular/central choice.
    """
    out = np.empty(len(scalars))
    for i, (dur, depth, mass, radius) in enumerate(np.asarray(scalars, float)):
        out[i] = period_from_duration(
            dur, mass, radius, k=np.sqrt(max(depth, 0.0)), b=b, ecc=ecc, contact="fwhm"
        )
    return out


class KNNPosterior:
    """Bayes-optimal scalar-only baseline: the Monte Carlo done exactly right.

    The training set is already a Monte Carlo sample from the joint prior
    p(P, M*, R*, k, b, e, omega) pushed through the forward model and the same
    measurement estimator used on real data. So the posterior p(log P | duration,
    depth, M*, R*) is obtained by rejection sampling - keep the training samples
    whose observables match the query - and k-nearest-neighbours is exactly that
    with an adaptive tolerance.

    This is the fair competitor the CNN-PINN must beat. It sees identical scalar
    inputs and carries identical priors, including the period prior, so any
    remaining advantage the network shows can only come from the light curve
    itself. The explicit (b, e, omega) marginalisation in 05/07 omits the period
    prior and is therefore *not* apples-to-apples.
    """

    def __init__(self, scalars, periods, k=250):
        self.k = k
        s = self._tx(scalars)
        self.mean, self.std = s.mean(0), s.std(0) + 1e-9
        self.train = (s - self.mean) / self.std
        self.log_p = np.log10(periods)

    @staticmethod
    def _tx(scalars):
        s = np.asarray(scalars, np.float64).copy()
        for j in (0, 1, 3):  # duration, depth, radius -> log
            s[:, j] = np.log10(np.clip(s[:, j], 1e-8, None))
        return s

    def predict(self, scalars, batch=64):
        """Return (median_period, lo_1sigma, hi_1sigma) arrays."""
        q = (self._tx(scalars) - self.mean) / self.std
        med = np.empty(len(q))
        lo = np.empty(len(q))
        hi = np.empty(len(q))
        for s in range(0, len(q), batch):
            chunk = q[s : s + batch]
            d2 = ((chunk[:, None, :] - self.train[None, :, :]) ** 2).sum(-1)
            idx = np.argpartition(d2, self.k, axis=1)[:, : self.k]
            neigh = self.log_p[idx]
            qs = np.percentile(neigh, [16, 50, 84], axis=1)
            lo[s : s + batch], med[s : s + batch], hi[s : s + batch] = qs[0], qs[1], qs[2]
        return 10.0**med, 10.0**lo, 10.0**hi


def metrics(pred, true):
    """Errors reported both linearly and in dex; dex is the meaningful one when
    the target spans three orders of magnitude."""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    ok = np.isfinite(pred) & (pred > 0) & np.isfinite(true) & (true > 0)
    pct = np.abs(pred[ok] - true[ok]) / true[ok] * 100.0
    dex = np.abs(np.log10(pred[ok] / true[ok]))
    ratio = pred[ok] / true[ok]
    return {
        "n": int(ok.sum()),
        "n_failed": int((~ok).sum()),
        "median_pct": float(np.median(pct)),
        "mean_pct": float(np.mean(pct)),
        "median_dex": float(np.median(dex)),
        "frac_within_10pct": float(np.mean(pct < 10)),
        "frac_within_25pct": float(np.mean(pct < 25)),
        "frac_within_factor_2": float(np.mean((ratio > 0.5) & (ratio < 2.0))),
        "median_bias_ratio": float(np.median(ratio)),
    }


def coverage(lo, hi, true):
    """Fraction of true periods inside the predicted interval, and median width
    in dex. A well-calibrated 1-sigma band should cover ~68%."""
    lo, hi, true = map(lambda a: np.asarray(a, float), (lo, hi, true))
    ok = np.isfinite(lo) & np.isfinite(hi) & (lo > 0)
    return {
        "coverage": float(np.mean((true[ok] >= lo[ok]) & (true[ok] <= hi[ok]))),
        "median_width_dex": float(np.median(np.log10(hi[ok] / lo[ok]))),
    }


def format_table(rows, headers):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows)
    return f"{line}\n{sep}\n{body}"
