"""Synthetic single-transit training set.

Design choices that differ from the v1-v3 generators, and why:

1. **Absolute-time light curves.** Every sample is a ``WINDOW_DAYS`` window
   sampled at ``N_POINTS``, identical to how real TESS data is extracted. The
   old generator used a self-normalised time axis, which destroyed the
   duration information the CNN was supposed to learn.

2. **Scalar features are *measured*, not injected.** Depth and duration come
   out of ``measure_transit`` applied to the simulated curve - the same
   estimator used on real light curves - so noise, limb darkening and grazing
   geometry bias the features the same way at train and test time.

3. **Physically consistent stars.** Mass and radius are drawn along a
   main-sequence relation with scatter plus an evolved tail, rather than
   independently. Duration depends on stellar *density*; independent sampling
   spends most of the training set on stars that cannot exist.

4. **Eccentricity is sampled, not ignored.** e ~ Beta(0.867, 3.03) (Kipping
   2013) with uniform omega. This is what makes the label noisy in a way the
   old generator hid: at fixed observables, the true period is genuinely
   uncertain by a factor of a few. The model is trained to report that.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .lightcurve import N_POINTS, WINDOW_DAYS, extract_window, measure_transit, simulate_transit


@dataclass
class SynthConfig:
    n_samples: int = 40_000
    seed: int = 42

    # Period floor must exceed the extraction window, otherwise a second transit
    # lands in the same array and the network can read the period straight off
    # the spacing. That is a different, far easier problem than the one this
    # project exists to solve, and including it inflates the synthetic metrics.
    period_min: float = 3.0          # days, log-uniform; > WINDOW_DAYS
    period_max: float = 1000.0
    mass_min: float = 0.15           # M_sun
    mass_max: float = 2.5
    k_min: float = 0.02              # Rp/R*, log-uniform (depth ~ 0.04% to 9%)
    k_max: float = 0.30
    b_max: float = 1.05              # includes grazing geometries
    evolved_fraction: float = 0.15
    radius_scatter_dex: float = 0.045

    ecc_a: float = 0.867             # Kipping (2013) Beta prior
    ecc_b: float = 3.03
    ecc_max: float = 0.85
    circular_fraction: float = 0.25  # fraction forced to e = 0

    noise_min: float = 1.0e-4        # per-cadence sigma (relative flux)
    noise_max: float = 3.0e-3
    wiggle_max: float = 6.0e-4       # residual detrending systematic

    window_days: float = WINDOW_DAYS
    n_points: int = N_POINTS
    max_duration_frac: float = 0.7   # reject transits filling most of the window
    min_transit_snr: float = 10.0    # integrated depth/sigma*sqrt(N_in), not per-cadence

    def to_dict(self):
        return asdict(self)


def _main_sequence_radius(mass, rng, cfg: SynthConfig):
    """R*/R_sun from M*/M_sun along the main sequence, with scatter and an
    evolved (subgiant/giant) tail."""
    radius = np.where(mass <= 1.0, mass**0.80, mass**0.57)
    radius *= 10.0 ** rng.normal(0.0, cfg.radius_scatter_dex, size=mass.shape)
    evolved = rng.random(mass.shape) < cfg.evolved_fraction
    radius = np.where(evolved, radius * rng.uniform(1.6, 3.5, size=mass.shape), radius)
    return np.clip(radius, 0.10, 6.0)


def _effective_temperature(mass):
    """Crude main-sequence T_eff, only used downstream for habitability."""
    return np.clip(5772.0 * mass**0.55, 2800.0, 9500.0)


def generate(cfg: SynthConfig | None = None, verbose: bool = True):
    """Generate the training set.

    Returns a dict of arrays:
      lc      (N, n_points)  windowed normalised flux
      scalars (N, 4)         [duration_fwhm_hours, depth, M*/M_sun, R*/R_sun]
      period  (N,)           true orbital period in days  (the label)
      meta    (N, 7)         [k, b, ecc, omega, teff, noise_sigma, t14_hours]
    """
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)

    lc_out, sc_out, p_out, meta_out = [], [], [], []
    attempts = 0
    batch = max(4096, cfg.n_samples // 4)

    while len(p_out) < cfg.n_samples:
        n = batch
        attempts += n

        period = np.exp(rng.uniform(np.log(cfg.period_min), np.log(cfg.period_max), n))
        mass = rng.uniform(cfg.mass_min, cfg.mass_max, n)
        radius = _main_sequence_radius(mass, rng, cfg)
        k = np.exp(rng.uniform(np.log(cfg.k_min), np.log(cfg.k_max), n))
        b = rng.uniform(0.0, cfg.b_max, n) * (1.0 + k)

        ecc = np.minimum(rng.beta(cfg.ecc_a, cfg.ecc_b, n), cfg.ecc_max)
        ecc = np.where(rng.random(n) < cfg.circular_fraction, 0.0, ecc)
        omega = rng.uniform(0.0, 2.0 * np.pi, n)

        u1 = rng.uniform(0.20, 0.60, n)
        u2 = rng.uniform(0.05, 0.40, n)
        sigma = np.exp(rng.uniform(np.log(cfg.noise_min), np.log(cfg.noise_max), n))
        t0 = rng.uniform(-0.05, 0.05, n) * cfg.window_days

        # Transit must actually occur.
        ok = b < (1.0 + k)
        if not np.any(ok):
            continue

        t_grid = np.linspace(-cfg.window_days / 2, cfg.window_days / 2, cfg.n_points)
        tt = np.broadcast_to(t_grid, (n, cfg.n_points))

        flux = simulate_transit(
            tt,
            t0[:, None],
            period[:, None],
            mass[:, None],
            radius[:, None],
            k[:, None],
            b[:, None],
            ecc[:, None],
            omega[:, None],
            u1[:, None],
            u2[:, None],
        )

        # Residual stellar variability / imperfect detrending, then white noise.
        wig_amp = rng.uniform(0.0, cfg.wiggle_max, n)[:, None]
        wig_per = rng.uniform(0.4, 6.0, n)[:, None]
        wig_ph = rng.uniform(0.0, 2 * np.pi, n)[:, None]
        flux = flux + wig_amp * np.sin(2 * np.pi * tt / wig_per + wig_ph)
        flux = flux + rng.normal(0.0, 1.0, flux.shape) * sigma[:, None]
        flux = flux / np.median(flux, axis=1, keepdims=True)

        max_dur_hr = cfg.max_duration_frac * cfg.window_days * 24.0
        for i in np.flatnonzero(ok):
            depth_m, dur_m, _ = measure_transit(t_grid, flux[i])
            if not np.isfinite(dur_m) or dur_m <= 0 or dur_m > max_dur_hr:
                continue
            # Standard integrated transit SNR. A per-cadence depth > 3 sigma cut
            # admits transits that are invisible in the array, whose measured
            # duration is set by the noise rather than the signal - a garbage
            # label the network is then trained to reproduce.
            cadence_hr = cfg.window_days * 24.0 / cfg.n_points
            n_in = max(dur_m / cadence_hr, 1.0)
            if depth_m / sigma[i] * np.sqrt(n_in) < cfg.min_transit_snr:
                continue

            lc_out.append(flux[i].astype(np.float32))
            sc_out.append([dur_m, depth_m, mass[i], radius[i]])
            p_out.append(period[i])
            meta_out.append(
                [k[i], b[i], ecc[i], omega[i], _effective_temperature(mass[i]), sigma[i], np.nan]
            )
            if len(p_out) >= cfg.n_samples:
                break

    data = {
        "lc": np.asarray(lc_out, np.float32),
        "scalars": np.asarray(sc_out, np.float32),
        "period": np.asarray(p_out, np.float32),
        "meta": np.asarray(meta_out, np.float32),
    }
    if verbose:
        acc = len(p_out) / attempts
        d = data["scalars"]
        print(f"[synth] kept {len(p_out):,} of {attempts:,} draws ({acc:.1%} pass detectability + window cuts)")
        print(f"[synth] period  {data['period'].min():8.2f} - {data['period'].max():8.2f} d")
        print(f"[synth] duration{d[:, 0].min():8.2f} - {d[:, 0].max():8.2f} h")
        print(f"[synth] depth   {d[:, 1].min() * 100:8.3f} - {d[:, 1].max() * 100:8.3f} %")
    return data


def save(data, path):
    np.savez_compressed(path, **data)


def load(path):
    with np.load(path) as f:
        return {key: f[key] for key in f.files}
