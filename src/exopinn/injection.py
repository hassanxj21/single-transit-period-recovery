"""Injection-recovery training set: synthetic transits injected into real TESS photometry.

Why this exists
---------------
A model trained purely on simulated light curves learned transit *shape*
features that beat a Bayes-optimal scalar baseline by 28% on synthetic data -
and then lost to a scalars-only model by 3.6x on real TESS targets. The shape
information it learned was a property of the generator, not of the sky.

Here the only simulated component is the transit signal itself. Everything else
- photon noise, spacecraft systematics, cadence, gaps, and crucially the
distortion imposed by detrending - comes from real TESS data, because the
transit is multiplied into the raw flux *before* the light curve is flattened.
The measurement path is byte-for-byte the one used at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import process_lightcurve
from .lightcurve import N_POINTS, WINDOW_DAYS, simulate_transit
from .synth import SynthConfig, _effective_temperature, _main_sequence_radius


@dataclass
class InjectionConfig(SynthConfig):
    # Injection uses the WHOLE contiguous chunk containing the injection epoch,
    # exactly as inference uses a whole sector. An earlier version cut a +/-6 day
    # window, which is a residual domain gap hiding inside otherwise-identical
    # code: Savitzky-Golay edge effects scale with filter length relative to
    # segment length, so a 12-day segment and a 27-day sector distort a long
    # transit differently even when the same function processes both.
    edge_margin_days: float = 1.0   # keep injections away from chunk edges
    min_chunk_days: float = 8.0     # skip chunks too short to detrend properly
    clear_of_real_dip_days: float = 3.0  # distance required from any masked feature
    gap_factor: float = 5.0         # break chunks where cadence gaps exceed this x median


def load_pool(path):
    """Return a list of (t, flux, excluded) arrays from the noise pool file."""
    with np.load(path) as z:
        keys = sorted((k for k in z.files if k.startswith("seg")),
                      key=lambda s: int(s[3:]))
        return [z[k] for k in keys]


def _draw_system(rng, cfg, n):
    period = np.exp(rng.uniform(np.log(cfg.period_min), np.log(cfg.period_max), n))
    mass = rng.uniform(cfg.mass_min, cfg.mass_max, n)
    radius = _main_sequence_radius(mass, rng, cfg)
    k = np.exp(rng.uniform(np.log(cfg.k_min), np.log(cfg.k_max), n))
    b = rng.uniform(0.0, cfg.b_max, n) * (1.0 + k)
    ecc = np.minimum(rng.beta(cfg.ecc_a, cfg.ecc_b, n), cfg.ecc_max)
    ecc = np.where(rng.random(n) < cfg.circular_fraction, 0.0, ecc)
    omega = rng.uniform(0.0, 2 * np.pi, n)
    u1 = rng.uniform(0.20, 0.60, n)
    u2 = rng.uniform(0.05, 0.40, n)
    return period, mass, radius, k, b, ecc, omega, u1, u2


def generate(pool, cfg: InjectionConfig | None = None, verbose=True):
    """Build the injected dataset. Same output schema as ``synth.generate``."""
    cfg = cfg or InjectionConfig()
    rng = np.random.default_rng(cfg.seed)

    lc_out, sc_out, p_out, meta_out = [], [], [], []
    attempts = 0
    max_dur_hr = cfg.max_duration_frac * cfg.window_days * 24.0
    cadence_hr = cfg.window_days * 24.0 / cfg.n_points

    while len(p_out) < cfg.n_samples:
        attempts += 1

        seg = pool[rng.integers(len(pool))]
        t_all, f_all, excl = seg[:, 0], seg[:, 1], seg[:, 2] > 0.5

        # Use a whole contiguous chunk, exactly as inference is handed a whole
        # sector. Cutting a fixed +/-N day window instead would make the filter
        # length a different fraction of the data than it is at inference, and
        # Savitzky-Golay edge effects scale with that fraction.
        dt = np.diff(t_all)
        med = np.median(dt) if len(dt) else 0.0
        if med <= 0:
            continue
        breaks = np.flatnonzero(dt > cfg.gap_factor * med) + 1
        chunks = [(a, b) for a, b in zip([0, *breaks.tolist()], [*breaks.tolist(), len(t_all)])
                  if b - a > 200 and (t_all[b - 1] - t_all[a]) >= cfg.min_chunk_days]
        if not chunks:
            continue
        a, b = chunks[rng.integers(len(chunks))]
        t = t_all[a:b].copy()
        f = f_all[a:b].copy()

        lo = t.min() + cfg.edge_margin_days
        hi = t.max() - cfg.edge_margin_days
        if hi <= lo:
            continue
        t0 = rng.uniform(lo, hi)

        # Stay well clear of any real dip: the detrending filter reaches several
        # days either side, so a nearby real feature would bias the trend.
        if excl[a:b][np.abs(t - t0) < cfg.clear_of_real_dip_days].any():
            continue

        period, mass, radius, k, b, ecc, omega, u1, u2 = _draw_system(rng, cfg, 1)
        period, mass, radius = period[0], mass[0], radius[0]
        k, b, ecc, omega, u1, u2 = k[0], b[0], ecc[0], omega[0], u1[0], u2[0]
        if b >= 1.0 + k:
            continue

        # --- inject into RAW flux, before any detrending ---------------------
        model = simulate_transit(t, t0, period, mass, radius, k, b, ecc, omega, u1, u2)
        f_inj = f * model

        # --- the real inference path, literally the same function ------------
        # t0 is passed as a hint because we know where we injected; at inference
        # it is found by search. Everything downstream is identical.
        res = process_lightcurve(t, f_inj, window_days=cfg.window_days,
                                 n_points=cfg.n_points, t0_hint=t0)
        if res is None:
            continue
        win, depth_m, dur_m, sigma = res["lc"], res["depth"], res["duration_hr"], res["scatter"]

        if not np.isfinite(dur_m) or dur_m <= 0 or dur_m > max_dur_hr:
            continue
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        n_in = max(dur_m / cadence_hr, 1.0)
        if depth_m / sigma * np.sqrt(n_in) < cfg.min_transit_snr:
            continue

        lc_out.append(win.astype(np.float32))
        sc_out.append([dur_m, depth_m, mass, radius])
        p_out.append(period)
        meta_out.append([k, b, ecc, omega, _effective_temperature(mass), sigma, t0])

        if verbose and len(p_out) % 2000 == 0:
            print(f"  [inject] {len(p_out):,}/{cfg.n_samples:,} "
                  f"({len(p_out)/attempts:.0%} of attempts kept)", flush=True)

    data = {
        "lc": np.asarray(lc_out, np.float32),
        "scalars": np.asarray(sc_out, np.float32),
        "period": np.asarray(p_out, np.float32),
        "meta": np.asarray(meta_out, np.float32),
    }
    if verbose:
        d = data["scalars"]
        print(f"[inject] kept {len(p_out):,} of {attempts:,} attempts ({len(p_out)/attempts:.1%})")
        print(f"[inject] period  {data['period'].min():8.2f} - {data['period'].max():8.2f} d")
        print(f"[inject] duration{d[:, 0].min():8.2f} - {d[:, 0].max():8.2f} h")
        print(f"[inject] depth   {d[:, 1].min()*100:8.3f} - {d[:, 1].max()*100:8.3f} %")
    return data
