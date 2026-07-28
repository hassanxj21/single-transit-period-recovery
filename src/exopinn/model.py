"""Hybrid CNN-PINN: light-curve shape + stellar scalars -> orbital period.

Two changes from v3 that matter more than the architecture:

* **The model predicts log10(period), and predicts its own uncertainty.**
  Period scales as duration cubed, so a symmetric error in linear period is
  meaningless and negative predictions become possible (v1 returned -10.88 d
  for TIC 438122862). Working in log space makes the target's error structure
  roughly homoscedastic and makes negative periods unrepresentable. The second
  output head is a per-sample log sigma trained with a Gaussian NLL, so the
  network reports a *range*. From a single transit, e and b are unobservable
  and the true period is genuinely uncertain by a factor of a few - a point
  estimate is not a defensible product, and the width is itself the result.

* **The physics loss uses the same duration estimator as the data pipeline**
  (full width at half depth, including the (1+k) planet-radius term).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .constants import R_SUN_AU
from .physics import semi_major_axis_torch, transit_duration_torch

SCALAR_NAMES = ["duration_fwhm_hr", "depth", "stellar_mass", "stellar_radius"]
LOG_SCALARS = [True, True, False, True]  # apply log10 before standardising


class Normalizer:
    """Feature/target standardisation, fitted on the training split only."""

    def __init__(self):
        self.sc_mean = self.sc_std = None
        self.lc_mean = self.lc_std = None
        self.y_mean = self.y_std = None

    @staticmethod
    def _tx(scalars):
        s = np.asarray(scalars, np.float64).copy()
        for j, use_log in enumerate(LOG_SCALARS):
            if use_log:
                s[:, j] = np.log10(np.clip(s[:, j], 1e-8, None))
        return s

    def fit(self, lc, scalars, period):
        s = self._tx(scalars)
        self.sc_mean, self.sc_std = s.mean(0), s.std(0) + 1e-9
        self.lc_mean, self.lc_std = float(lc.mean()), float(lc.std() + 1e-12)
        y = np.log10(period)
        self.y_mean, self.y_std = float(y.mean()), float(y.std() + 1e-12)
        return self

    def transform_lc(self, lc):
        return ((lc - self.lc_mean) / self.lc_std).astype(np.float32)

    def transform_scalars(self, scalars):
        return ((self._tx(scalars) - self.sc_mean) / self.sc_std).astype(np.float32)

    def transform_y(self, period):
        return ((np.log10(period) - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, z):
        """Normalised -> log10(period). Works for numpy or torch."""
        return z * self.y_std + self.y_mean

    def state_dict(self):
        return {
            "sc_mean": self.sc_mean, "sc_std": self.sc_std,
            "lc_mean": self.lc_mean, "lc_std": self.lc_std,
            "y_mean": self.y_mean, "y_std": self.y_std,
        }

    @classmethod
    def from_state_dict(cls, d):
        n = cls()
        n.sc_mean = np.asarray(d["sc_mean"]); n.sc_std = np.asarray(d["sc_std"])
        n.lc_mean = float(d["lc_mean"]); n.lc_std = float(d["lc_std"])
        n.y_mean = float(d["y_mean"]); n.y_std = float(d["y_std"])
        return n


class HybridCNNPINN(nn.Module):
    def __init__(self, n_points: int = 128, n_scalars: int = 4, dropout: float = 0.10,
                 use_lightcurve: bool = True, min_log_sigma: float = -6.0):
        super().__init__()
        # use_lightcurve=False drops the CNN branch entirely, leaving a
        # scalars-only model. That ablation is what separates "the CNN reads
        # impact parameter off the transit shape" from "the network learned the
        # period prior" - the two explanations for beating a scalar baseline.
        self.use_lightcurve = use_lightcurve
        self.min_log_sigma = min_log_sigma
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
        )
        cnn_dim = 64 * (n_points // 8) if use_lightcurve else 0
        self.scalar_net = nn.Sequential(
            nn.Linear(n_scalars, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(cnn_dim + 64, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2),  # [mu, log_sigma] of normalised log10(period)
        )

    def forward(self, lc, scalars):
        h = self.scalar_net(scalars)
        if self.use_lightcurve:
            h = torch.cat([self.cnn(lc), h], dim=1)
        out = self.head(h)
        mu = out[:, 0:1]
        # The floor is not cosmetic: with an unbounded lower clamp the NLL drives
        # sigma toward zero on training points, which is exactly how the 300-epoch
        # run reached a val NLL of +6.07 while its train NLL was still falling.
        log_sigma = torch.clamp(out[:, 1:2], self.min_log_sigma, 3.0)
        return mu, log_sigma


def gaussian_nll(mu, log_sigma, target):
    """Heteroscedastic negative log-likelihood (constants dropped)."""
    inv_var = torch.exp(-2.0 * log_sigma)
    return torch.mean(0.5 * inv_var * (target - mu) ** 2 + log_sigma)


# Natural-log tolerance on the duration residual. The circular-orbit model
# cannot be an equality constraint: eccentricity rescales duration by
# sqrt(1-e^2)/(1+e sin w), which spans 0.73x-1.36x for e = 0.3 (|ln| = 0.31),
# and limb darkening biases a half-depth measurement low by a further ~6%.
# Inside this band the physics term is silent; outside it, it pulls hard.
DURATION_LOG_TOL = 0.35


def physics_loss(period_days, duration_hr, mass, radius, depth, tol=DURATION_LOG_TOL):
    """Kepler + transit-geometry constraint, as a tolerance band plus a hard floor.

    Two terms:
      * a one-sided log-space duration residual - the Kepler constraint
        expressed through the observable the network actually sees, penalised
        only beyond what eccentricity and limb darkening could plausibly explain;
      * a hinge penalty when the predicted period implies an orbit inside the
        star (a < R*), which is what let v1 emit negative and sub-stellar periods.
    """
    k = torch.sqrt(torch.clamp(depth, min=1e-8))
    b = torch.zeros_like(k)

    model_dur = transit_duration_torch(period_days, mass, radius, k, b, contact="fwhm")
    residual = torch.log(torch.clamp(model_dur, min=1e-4)) - torch.log(torch.clamp(duration_hr, min=1e-4))
    excess = torch.relu(torch.abs(residual) - tol)

    a = semi_major_axis_torch(period_days, mass)
    a_over_r = a / (radius * R_SUN_AU)
    hinge = torch.relu(1.0 - a_over_r) ** 2

    return torch.mean(excess**2) + 10.0 * torch.mean(hinge)
