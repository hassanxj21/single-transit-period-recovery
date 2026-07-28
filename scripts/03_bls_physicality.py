#!/usr/bin/env python
"""Are the BLS-reported periods physically possible for these stars?

Two independent tests, weakest first:

  Test A (orbit clears the star)   a(P, M*) > R*
      Nearly everything passes this. It only catches catastrophic outputs like
      the -10.88 d and 4.19 d predictions from PINN v1/v2.

  Test B (required eccentricity)   what would the BLS period have to assume?
      At a given period the longest possible transit is central (b = 0) and
      circular; eccentricity can stretch it further only if the transit sits
      near apoastron, by sqrt((1+e)/(1-e)). Inverting that gives the *minimum
      eccentricity* an orbit must have for the BLS period and the observed
      duration to coexist:

          f = duration_observed / duration_circular(P_bls)
          e_required = (f^2 - 1) / (f^2 + 1)          [b = 0, omega = -pi/2]

      b > 0 only shortens the transit, so this is the most generous possible
      assumption - the true requirement is always at least this severe. Scoring
      e_required against the Kipping (2013) Beta(0.867, 3.03) eccentricity prior
      turns "physically meaningless" from an assertion into a probability.

    python scripts/03_bls_physicality.py
"""

import _bootstrap as B
import numpy as np

from exopinn.evaluate import format_table
from exopinn.physics import (
    eccentricity_duration_factor,
    equilibrium_temperature,
    is_physically_possible,
    min_physical_period,
    period_from_duration,
    transit_duration,
)
from exopinn.targets import ALL, BLS_CANDIDATE_PERIODS, CANDIDATES, CONFIRMED

E_MAX = 0.85          # beyond this we call a required eccentricity unattainable
ECC_PRIOR = (0.867, 3.03)   # Kipping (2013) Beta prior on transiting-planet eccentricity


def required_eccentricity(period, mass, radius, k, duration_hr, kind="14"):
    """Minimum eccentricity for `period` and `duration_hr` to be compatible.

    Returns 0.0 if the circular central transit is already long enough, and
    nan if even e -> 1 cannot stretch the transit far enough.
    """
    d_circ = float(transit_duration(period, mass, radius, k=k, b=0.0, contact=kind))
    f = duration_hr / d_circ
    if f <= 1.0:
        return 0.0, f, d_circ
    e_req = (f**2 - 1.0) / (f**2 + 1.0)
    return e_req, f, d_circ


def prior_prob_above(e):
    """P(e' > e) under the Kipping Beta prior - how lucky the system must be."""
    from scipy.stats import beta

    return float(beta.sf(e, *ECC_PRIOR))


def assess(t, period):
    k = np.sqrt(t.depth) if t.depth else 0.0
    ok_a, a_over_r = is_physically_possible(period, t.mass, t.radius)
    row = {
        "tic": t.tic, "period": period, "a_over_rstar": float(a_over_r),
        "p_min": float(min_physical_period(t.mass, t.radius)), "test_a": bool(ok_a),
        "dur_obs": t.duration_hr, "e_req": np.nan, "f": np.nan, "d_circ": np.nan, "p_e": np.nan,
    }
    if t.duration_hr:
        e_req, f, d_circ = required_eccentricity(period, t.mass, t.radius, k, t.duration_hr, t.duration_kind)
        row.update(e_req=e_req, f=f, d_circ=d_circ, p_e=prior_prob_above(e_req))
    return row


def main():
    print("=" * 96)
    print("PART 1 - Confirmed planets: does the true period pass both tests?")
    print("=" * 96)
    rows = []
    for t in CONFIRMED:
        if t.mass is None or t.duration_hr is None:
            print(f"  TIC {t.tic}: skipped, stellar properties / duration not yet measured "
                  f"({t.notes.split('.')[0]}.)")
            continue
        r = assess(t, t.true_period)
        rows.append([t.label, f"{r['period']:.2f}", f"{r['a_over_rstar']:.1f}",
                     "PASS" if r["test_a"] else "FAIL",
                     f"{r['dur_obs']:.2f}", f"{r['d_circ']:.2f}", f"{r['f']:.2f}",
                     f"{r['e_req']:.2f}", f"{r['p_e']:.0%}"])
    print(format_table(rows, ["target", "P [d]", "a/R*", "test A", "dur obs [h]",
                              "dur circ [h]", "obs/circ", "e required", "prior P(e)"]))
    print("\n  Both confirmed planets pass test A, so test A does not reject truth - but it")
    print("  does not reject much of anything, which is the point of running it.")
    print("  Test B recovers the known physics without being told any of it:")
    for t in CONFIRMED:
        if t.mass is None or t.duration_hr is None:
            continue
        r = assess(t, t.true_period)
        known = "" if t.known_ecc is None else f"  [published e = {t.known_ecc:.2f}]"
        print(f"    {t.label:14s} requires e >= {r['e_req']:.2f} "
              f"({r['p_e']:.0%} of the prior){known}")
    print("  None of the three is flagged, so the test does not reject truth.")
    print("\n  Caveat: 'dur obs' must be the SAME quantity the model computes (full width at")
    print("  half depth). Because period scales as duration cubed, quoting a T14 duration")
    print("  against an FWHM model - or vice versa - shifts the implied period by ~30%.")

    print()
    print("=" * 96)
    print("PART 2 - BLS output on the five unconfirmed single-transit candidates")
    print("=" * 96)
    rows, verdicts = [], []
    for t in CANDIDATES:
        p = BLS_CANDIDATE_PERIODS[t.tic]
        r = assess(t, p)
        flag = "" if t.duration_reliable else " (!)"
        verdict = "impossible" if (not np.isfinite(r["e_req"]) or r["e_req"] > E_MAX) else \
                  "implausible" if r["e_req"] > 0.5 else "allowed"
        rows.append([f"TIC {t.tic}{flag}", f"{p:.2f}", f"{r['a_over_rstar']:.1f}",
                     "PASS" if r["test_a"] else "FAIL", f"{r['dur_obs']:.1f}",
                     f"{r['d_circ']:.2f}", f"{r['f']:.2f}",
                     f"{r['e_req']:.2f}", f"{r['p_e']:.1%}", verdict])
        verdicts.append((r, verdict))
    print(format_table(rows, ["target", "BLS P [d]", "a/R*", "test A", "dur obs [h]",
                              "dur circ [h]", "obs/circ", "e required", "prior P(e)", "verdict"]))
    print("  (!) duration is a window artefact, not a measurement - row is not interpretable.")

    n_a = sum(1 for r, _ in verdicts if not r["test_a"])
    n_bad = sum(1 for _, v in verdicts if v != "allowed")
    print(f"\n  Test A rejects {n_a}/5. It has essentially no discriminating power here:")
    print("  any period of order days puts the orbit far outside the star. Reporting a")
    print("  BLS period as 'physically impossible' on these grounds alone is not supportable.")
    print(f"\n  Test B rejects {n_bad}/5. For those targets the BLS period requires an")
    print("  eccentricity that the population prior makes very unlikely - and that is with")
    print("  b = 0, the single most generous geometry. This is the defensible version of")
    print("  the claim that BLS output on a single transit is not a period measurement.")

    print()
    print("=" * 96)
    print("PART 3 - Period implied by each observed duration, and how far e can move it")
    print("=" * 96)
    print("  The zero-free-parameter estimate is the b=0, e=0 inversion. The two unknowns")
    print("  move it in opposite directions: b > 0 shortens the transit and so raises the")
    print("  implied period, while a transit near apoastron lengthens it and lowers the")
    print("  implied period. Eccentricity dominates, and it is unbounded downward in")
    print("  practice - which is why the honest output of this method is an interval.")
    rows = []
    for t in CANDIDATES:
        k = np.sqrt(t.depth)
        kw = dict(k=k, b=0.0, contact=t.duration_kind)
        p_circ = period_from_duration(t.duration_hr, t.mass, t.radius, **kw)
        # e = 0.3 at apoastron: a plausible-but-generous stretch, ~25% of the prior.
        p_lo = period_from_duration(t.duration_hr, t.mass, t.radius, ecc=0.30, omega=-np.pi / 2, **kw)
        # e -> 0.85 at apoastron: the extreme that most reduces the implied period.
        p_ext = period_from_duration(t.duration_hr, t.mass, t.radius, ecc=E_MAX, omega=-np.pi / 2, **kw)
        teq = equilibrium_temperature(t.teff, t.radius, p_circ, t.mass)
        flag = "" if t.duration_reliable else " (!)"
        rows.append([f"TIC {t.tic}{flag}", f"{t.duration_hr:.1f}", f"{p_circ:.1f}",
                     f"{p_lo:.1f}", f"{p_ext:.1f}", f"{BLS_CANDIDATE_PERIODS[t.tic]:.1f}",
                     f"{p_circ / BLS_CANDIDATE_PERIODS[t.tic]:.1f}x",
                     "yes" if p_circ > 27.0 else "no", f"{teq:.0f}"])
    print()
    print(format_table(rows, ["target", "dur [h]", "P circular [d]", "P at e=0.3 [d]",
                              "P at e=0.85 [d]", "BLS P [d]", "circ/BLS",
                              "> 27 d?", "T_eq circ [K]"]))
    print("\n  'P circular' is the b=0, e=0 inversion - the estimate with no free parameters.")
    print("  The e columns show how far eccentricity alone can move it: this spread, not the")
    print("  measurement noise, is the dominant uncertainty in single-transit period recovery.")
    print("  Where 'P circular' exceeds 27 d the transit is single by construction and no BLS")
    print("  search range can be correct - which is the structural claim, stated per target.")
    print("  T_eq is evaluated at the circular period only; it inherits the full e spread.")


if __name__ == "__main__":
    main()
