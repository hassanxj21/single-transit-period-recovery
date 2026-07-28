#!/usr/bin/env python
"""TIC 65910228 / NGTS-38 b - the decisive single-transit case, from published
parameters only. Runs without MAST.

Three things this establishes:

1. **BLS cannot recover this period at any search range, for a structural
   reason.** The period is 180.53 d and the sector baseline is 27 d, so the
   planet completes 0.15 of an orbit while TESS watches. With one transit and
   no second event, the BLS likelihood is flat in period above the baseline -
   every trial period longer than the data predicts exactly one transit and
   fits identically well. There is no maximum to find. Any peak the
   periodogram reports is noise in the out-of-transit scatter, and restricting
   the grid just moves the reported answer to wherever the grid ends.

2. **The physics inversion works but is degenerate.** Impact parameter and
   eccentricity are both unobservable from a single transit and pull the
   implied period in opposite directions. For this planet they nearly cancel.

3. **The correct output is a posterior, not a number.** Marginalising over
   population priors for b and e gives an interval - and that interval is what
   a calibrated CNN-PINN should reproduce.

    python scripts/05_ngts38_offline.py
"""

import _bootstrap as B
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

from exopinn.constants import SECTOR_BASELINE_DAYS
from exopinn.evaluate import format_table
from exopinn.physics import equilibrium_temperature, period_from_duration, transit_duration
from exopinn.targets import ALL

ECC_PRIOR = (0.867, 3.03)
N_MC = 20_000
SEED = 7


def main():
    t = ALL[65910228]
    k = t.known_k
    print("=" * 88)
    print(f"{t.label} (TIC {t.tic}) - published parameters (NASA Exoplanet Archive)")
    print("=" * 88)
    print(f"  P = {t.true_period:.3f} d    M* = {t.mass} Msun    R* = {t.radius} Rsun    "
          f"Teff = {t.teff} K")
    print(f"  T14 = {t.duration_hr} h    k = Rp/R* = {k}    b = {t.known_b}    e = {t.known_ecc}")

    # ---------------------------------------------------------------- 1. BLS
    print()
    print("-" * 88)
    print("1. Why BLS cannot work here, independent of search range")
    print("-" * 88)
    cycles = SECTOR_BASELINE_DAYS / t.true_period
    print(f"  Orbital cycles completed during one 27-day sector: {cycles:.3f}")
    print(f"  Transits available to phase-fold: 1")
    print()
    rows = []
    for lo, hi in [(0.5, 27.0), (0.5, 100.0), (20.0, 100.0), (50.0, 300.0), (100.0, 400.0)]:
        best_possible = min(max(t.true_period, lo), hi)   # closest the grid can get
        err = abs(best_possible - t.true_period) / t.true_period * 100
        contains = lo <= t.true_period <= hi
        rows.append([f"{lo:g}-{hi:g}", "yes" if contains else "no",
                     f"{best_possible:.2f}", f"{err:.1f}%",
                     "grid cannot reach it" if not contains else "grid can reach it, but see below"])
    print(format_table(rows, ["search range [d]", "contains truth?", "best reachable [d]",
                              "floor on error", "note"]))
    print()
    print("  The first four rows are the structural failure: the answer is outside the grid,")
    print("  so the error is bounded below by the grid edge no matter how the data looks.")
    print("  The last row is the subtler failure. Even when the grid contains 180.5 d, one")
    print("  transit gives no periodicity information: every trial period longer than the")
    print("  27-day baseline predicts exactly one transit in the window and therefore fits")
    print("  the data equally well. The periodogram is flat, and argmax over a flat surface")
    print("  returns noise. BLS needs a second transit, not a better grid.")

    # ---------------------------- 2. the degeneracy, at the published duration
    print()
    print("-" * 88)
    print("2. Period implied by the observed 14.86 h transit, under different assumptions")
    print("-" * 88)
    cases = [
        ("circular, central (b=0, e=0)", 0.0, 0.0, np.pi / 2),
        ("circular, published b=0.854", t.known_b, 0.0, np.pi / 2),
        ("b=0, e=0.31 at apoastron", 0.0, t.known_ecc, -np.pi / 2),
        ("published b and e (truth)", t.known_b, t.known_ecc, -np.pi / 2),
    ]
    rows = []
    for lab, b, e, om in cases:
        p = period_from_duration(t.duration_hr, t.mass, t.radius, k=k, b=b, ecc=e,
                                 omega=om, contact="14")
        teq = equilibrium_temperature(t.teff, t.radius, p, t.mass)
        rows.append([lab, f"{p:.1f}", f"{p / t.true_period:.2f}x", f"{teq:.0f}"])
    print(format_table(rows, ["assumption", "implied P [d]", "vs truth", "T_eq [K]"]))
    p_circ, p_b, p_e = (float(r[1]) for r in (rows[0], rows[1], rows[2]))
    print()
    print(f"  Acting alone, the published b raises the implied period by {p_b / p_circ:.1f}x and the")
    print(f"  published e lowers it by {p_e / p_circ:.2f}x. Here they nearly cancel, which is luck,")
    print(f"  not method: the circular estimate is off by {t.true_period / p_circ:.1f}x. Even so it lands")
    print("  closer to the truth than any BLS answer restricted to the sector baseline.")

    # ------------------------------------- 3. marginalise: the physics posterior
    print()
    print("-" * 88)
    print("3. Physics-only posterior: marginalising over the b and e priors")
    print("-" * 88)
    rng = np.random.default_rng(SEED)
    b_s = rng.uniform(0.0, 1.0 + k, N_MC)
    e_s = np.minimum(beta_dist.rvs(*ECC_PRIOR, size=N_MC, random_state=rng), 0.9)
    om_s = rng.uniform(0.0, 2 * np.pi, N_MC)

    periods = np.full(N_MC, np.nan)
    for i in range(N_MC):
        periods[i] = period_from_duration(t.duration_hr, t.mass, t.radius, k=k,
                                          b=b_s[i], ecc=e_s[i], omega=om_s[i], contact="14")
    good = np.isfinite(periods) & (periods > 0)
    p_ok = periods[good]
    q = np.percentile(p_ok, [2.5, 16, 50, 84, 97.5])

    print(f"  {good.sum():,} of {N_MC:,} sampled geometries admit a transit")
    print(f"  median            {q[2]:8.1f} d")
    print(f"  68% interval      {q[1]:8.1f} - {q[3]:.1f} d   (factor {q[3]/q[1]:.1f} wide)")
    print(f"  95% interval      {q[0]:8.1f} - {q[4]:.1f} d   (factor {q[4]/q[0]:.1f} wide)")
    print(f"  truth {t.true_period:.1f} d sits at percentile "
          f"{100 * np.mean(p_ok < t.true_period):.1f}")
    print()
    print("  This is the honest answer from one transit: a period known to within a factor")
    print("  of a few, with the truth comfortably inside. It is not a precise measurement,")
    print("  and no method that uses only a single transit can make it one - but it is")
    print("  enough to say the planet is not a hot Jupiter, and enough to schedule a")
    print("  follow-up window. That is the actual scientific product.")

    teq_s = equilibrium_temperature(t.teff, t.radius, p_ok, t.mass)
    tq = np.percentile(teq_s, [16, 50, 84])
    hz = np.mean((teq_s >= 200) & (teq_s <= 320))
    print(f"\n  Equilibrium temperature {tq[1]:.0f} K  (68%: {tq[0]:.0f}-{tq[2]:.0f} K), "
          f"P(habitable zone) = {hz:.1%}")
    print(f"  T_eq scales as P^(-1/3), so the factor-{q[3]/q[1]:.1f} period interval compresses to a")
    print(f"  factor-{tq[2]/tq[0]:.1f} temperature interval. Habitability screening tolerates")
    print("  single-transit precision far better than the period itself does, which is the")
    print("  strongest argument for the whole approach.")
    print()
    print("  Where the width comes from, and why the CNN branch exists: the long tail is")
    print("  driven by near-grazing geometries (b -> 1+k), which need a very long period to")
    print("  produce a 14.86 h chord. But a grazing transit is V-shaped and shallow-sloped -")
    print("  the light curve SHAPE constrains b directly, and shape is exactly what a scalar")
    print("  duration-and-depth inversion throws away. Narrowing this interval by reading b")
    print("  off the ingress profile is the one thing the CNN can do that the analytic")
    print("  inversion cannot, and it is the claim the architecture should be tested on.")

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    axes[0].hist(np.log10(p_ok), bins=80, color="#3b6ea5", alpha=0.85, density=True)
    axes[0].axvline(np.log10(t.true_period), color="crimson", lw=2, label=f"truth {t.true_period:.0f} d")
    axes[0].axvspan(np.log10(q[1]), np.log10(q[3]), color="k", alpha=0.12, label="68%")
    axes[0].axvline(np.log10(SECTOR_BASELINE_DAYS), color="darkorange", ls="--", lw=1.5,
                    label="27 d sector baseline")
    axes[0].set_xlabel("log10 implied period [d]"); axes[0].set_ylabel("density")
    axes[0].set_title("Physics-only posterior\n(marginalised over b, e, omega)")
    axes[0].legend(fontsize=8)

    bs = np.linspace(0.0, 1.0 + k - 1e-3, 160)
    es = np.linspace(0.0, 0.8, 160)
    BB, EE = np.meshgrid(bs, es)
    grid = np.array([[period_from_duration(t.duration_hr, t.mass, t.radius, k=k, b=bb,
                                           ecc=ee, omega=-np.pi / 2, contact="14")
                      for bb in bs] for ee in es])
    im = axes[1].pcolormesh(BB, EE, np.log10(grid), cmap="viridis", shading="auto")
    cs = axes[1].contour(BB, EE, np.log10(grid), levels=[np.log10(t.true_period)],
                         colors="crimson", linewidths=2)
    axes[1].clabel(cs, fmt={np.log10(t.true_period): "truth"})
    axes[1].plot(t.known_b, t.known_ecc, "w*", ms=16, mec="k", label="published solution")
    axes[1].set_xlabel("impact parameter b"); axes[1].set_ylabel("eccentricity e")
    axes[1].set_title("Implied period from a fixed 14.86 h transit\n"
                      "(colour = log10 P; omega fixed at -90 deg)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.colorbar(im, ax=axes[1], label="log10 P [d]")

    axes[2].hist(teq_s, bins=80, color="#a5533b", alpha=0.85, density=True)
    axes[2].axvspan(200, 320, color="green", alpha=0.15, label="habitable zone")
    axes[2].axvline(equilibrium_temperature(t.teff, t.radius, t.true_period, t.mass),
                    color="crimson", lw=2, label="truth")
    axes[2].set_xlabel("equilibrium temperature [K]")
    axes[2].set_title(f"T_eq posterior\n68%: {tq[0]:.0f}-{tq[2]:.0f} K")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(B.FIGURES / "ngts38b_degeneracy.png", dpi=150)
    print(f"\n  figure -> {B.FIGURES / 'ngts38b_degeneracy.png'}")


if __name__ == "__main__":
    main()
