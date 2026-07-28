#!/usr/bin/env python
"""Pre-flight gate. The head-to-head must not launch unless this passes.

  A. Ephemeris integrity - every CSV row's pl_name must match the archive row
     the ephemeris came from, exactly. The tic_id-only merge silently gave five
     targets in multi-planet systems another planet's ephemeris; only a name
     assertion catches that class of error.
  B. Known-answer check - TOI-2134 c must select S52 (n=0) and reproduce
     depth 1.008%, FWHM 4.27 h, SNR ~182. It selected S80 (n=8) under the
     previous ranking and measured 0.053% / 2.20 h / SNR 2.5, a factor of 19
     in depth from sector choice alone.

    python scripts/15_preflight.py
"""

import csv
import sys
import warnings

import _bootstrap as B
import numpy as np

from exopinn.data import prepare_target, sector_with_transit
from exopinn.targets import CONFIRMED, EXPANDED

warnings.filterwarnings("ignore")

FAILURES = []


def check_a():
    print("=" * 92)
    print("A. EPHEMERIS INTEGRITY - pl_name assertion")
    print("=" * 92)
    path = B.DATA / "expanded_targets.csv"
    rows = list(csv.DictReader(path.open()))
    print(f"  {'CSV pl_name':20s} {'archive pl_name':20s} {'TIC':>11} {'tranmid (BJD)':>16}  match")
    bad = 0
    for r in rows:
        csv_name = r["pl_name"]
        arch_name = r.get("archive_pl_name", "")
        ok = (csv_name == arch_name) and bool(r.get("pl_tranmid"))
        if not ok:
            bad += 1
        print(f"  {csv_name:20s} {arch_name:20s} {r['tic']:>11} "
              f"{r.get('pl_tranmid', '-'):>16}  {'OK' if ok else '*** MISMATCH ***'}")
    if bad:
        FAILURES.append(f"A: {bad} ephemeris name mismatches")
    print(f"\n  {len(rows) - bad}/{len(rows)} exact name matches"
          f"{' - FAIL' if bad else ' - PASS'}")
    return bad == 0


def check_b():
    print("\n" + "=" * 92)
    print("B. KNOWN-ANSWER CHECK - TOI-2134 c")
    print("=" * 92)
    tgt = [t for t in EXPANDED if t.name == "TOI-2134 c"]
    if not tgt:
        FAILURES.append("B: TOI-2134 c not in target list")
        print("  TOI-2134 c not found - FAIL")
        return False
    t = tgt[0]
    expect = {"sector": 52, "n": 0, "depth_pct": 1.008, "fwhm": 4.27, "snr": 182.0}
    print(f"  expected: S{expect['sector']} n={expect['n']} depth={expect['depth_pct']}% "
          f"FWHM={expect['fwhm']}h SNR~{expect['snr']:.0f}")

    hit = sector_with_transit(t.tic, t.tranmid, t.true_period, verbose=True)
    if hit is None:
        FAILURES.append("B: no sector selected for TOI-2134 c")
        print("  no sector selected - FAIL")
        return False

    p = prepare_target(t.tic, sector=hit["sector"], search_window=hit["window"],
                       mass=t.mass, radius=t.radius, verbose=False)
    if "error" in p:
        FAILURES.append(f"B: prepare_target failed ({p['error']})")
        print(f"  prepare_target failed: {p['error']} - FAIL")
        return False

    got = {"sector": hit["sector"], "n": hit["n_cycles"],
           "depth_pct": p["depth"] * 100, "fwhm": p["duration_hr"], "snr": p["snr"]}
    print(f"  measured: S{got['sector']} n={got['n']} depth={got['depth_pct']:.3f}% "
          f"FWHM={got['fwhm']:.2f}h SNR={got['snr']:.1f}")

    ok = (got["sector"] == expect["sector"] and got["n"] == expect["n"]
          and abs(got["depth_pct"] - expect["depth_pct"]) < 0.05
          and abs(got["fwhm"] - expect["fwhm"]) < 0.2
          and abs(got["snr"] - expect["snr"]) / expect["snr"] < 0.10)
    if not ok:
        FAILURES.append("B: TOI-2134 c does not reproduce the known answer")
    print(f"  {'PASS' if ok else '*** FAIL ***'}")
    return ok


def expectations():
    print("\n" + "=" * 92)
    print("C. EXPECTATIONS STATED BEFORE THE RUN")
    print("=" * 92)
    n = len(CONFIRMED) + len(EXPANDED)
    print(f"  targets attempted            {n}")
    print(f"  expected runtime             ~20-30 min (~40 s/target, one download each)")
    print("  SNR gate                     integrated SNR >= 7.5")
    print("    derivation: sigma_T/T = sqrt(2)*sqrt(tau/T)/Q with tau/T ~ 0.09-0.10")
    print("    -> sigma_T/T ~ 0.45/Q; at Q=7.5 that is 6.0% duration precision,")
    print("       and since P ~ T^3, 18% period precision. Conventional detection")
    print("       is Q ~ 7.1 (18.9%); 7.5 is the round number just above it.")
    print("\n  predicted SNR-gate failures (from earlier measurements):")
    for name, pred in [("TOI-4562 b", 2.9), ("TOI-2134 c (old S80)", 2.5),
                       ("HD 56414 b", "no transit found"),
                       ("TOI-5110 b", 5.7), ("HD 28109 d", 5.0)]:
        print(f"    {name:24s} predicted integrated SNR {pred}")
    print("\n  If the run produces ZERO SNR exclusions, treat that as a gate bug,")
    print("  not as good news.")


def main():
    ok_a = check_a()
    ok_b = check_b()
    expectations()

    print("\n" + "=" * 92)
    if FAILURES:
        print("PRE-FLIGHT FAILED - do not launch")
        for f in FAILURES:
            print(f"  {f}")
        print("=" * 92)
        sys.exit(1)
    print("PRE-FLIGHT PASSED - safe to launch the head-to-head")
    print("=" * 92)


if __name__ == "__main__":
    main()
