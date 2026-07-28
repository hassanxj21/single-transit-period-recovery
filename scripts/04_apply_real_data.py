#!/usr/bin/env python
"""Apply BLS, the analytic inversion and the CNN-PINN to real TESS targets.

    python scripts/04_apply_real_data.py                 # every target
    python scripts/04_apply_real_data.py --tic 65910228  # one target
    python scripts/04_apply_real_data.py --no-bls        # skip the slow part

Requires MAST (mast.stsci.edu). If MAST is unreachable the script says so and
exits rather than silently producing nothing; use 05_ngts38_offline.py for the
parts of the NGTS-38 b analysis that need only published parameters.
"""

import argparse
import json

import _bootstrap as B
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exopinn import data as D
from exopinn.bls import range_dependency
from exopinn.evaluate import format_table
from exopinn.physics import equilibrium_temperature, period_from_duration
from exopinn.targets import CANDIDATES, CONFIRMED
from exopinn.train import load_model, predict

# Search ranges for the range-dependency experiment. The first spans the sector,
# the rest deliberately exclude the true period for the confirmed targets.
BLS_RANGES = [(0.5, 27.0), (0.5, 100.0), (20.0, 100.0), (50.0, 300.0)]


def check_mast():
    import socket

    try:
        socket.create_connection(("mast.stsci.edu", 443), timeout=12).close()
        return True
    except OSError as exc:
        print(f"[04] cannot reach mast.stsci.edu: {exc}")
        print("[04] MAST is required for light-curve download. Nothing was computed.")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int, default=None)
    ap.add_argument("--model", default=str(B.MODELS / "cnn_pinn" / "cnn_pinn.pt"))
    ap.add_argument("--no-bls", action="store_true")
    args = ap.parse_args()

    if not check_mast():
        raise SystemExit(2)

    targets = CONFIRMED + CANDIDATES
    if args.tic:
        targets = [t for t in targets if t.tic == args.tic] or targets

    model, norm, _ = load_model(args.model)
    print(f"[04] loaded {args.model}\n")

    results = []
    for t in targets:
        print(f"--- {t.label} ---")
        try:
            prep = D.prepare_target(t.tic, sector=t.sector, search_window=t.window,
                                    mass=t.mass, radius=t.radius)
        except Exception as exc:
            print(f"    download/prepare failed: {exc}\n")
            continue
        if "error" in prep:
            print(f"    {prep['error']}\n")
            continue
        if "scalars" not in prep:
            print("    missing stellar mass/radius, cannot run the model\n")
            continue

        row = {"tic": t.tic, "label": t.label, "true_period": t.true_period,
               "depth_measured": prep["depth"], "duration_measured_fwhm": prep["duration_hr"],
               "snr": prep["snr"], "provenance": prep["provenance"]}

        # Analytic inversion. Measured duration is FWHM by construction here.
        k = np.sqrt(max(prep["depth"], 0.0))
        row["analytic_circular"] = period_from_duration(
            prep["duration_hr"], prep["mass"], prep["radius"], k=k, b=0.0, contact="fwhm")
        row["analytic_b070"] = period_from_duration(
            prep["duration_hr"], prep["mass"], prep["radius"], k=k, b=0.70, contact="fwhm")
        row["analytic_e030"] = period_from_duration(
            prep["duration_hr"], prep["mass"], prep["radius"], k=k, b=0.0,
            ecc=0.30, omega=-np.pi / 2, contact="fwhm")

        # CNN-PINN
        p, lo, hi = predict(model, norm, prep["lc"][None, :], prep["scalars"])
        row["pinn"], row["pinn_lo"], row["pinn_hi"] = float(p[0]), float(lo[0]), float(hi[0])

        teff = prep.get("teff") or t.teff
        if teff:
            row["teq_pinn"] = float(equilibrium_temperature(teff, prep["radius"], p[0], prep["mass"]))

        # BLS across several search ranges
        if not args.no_bls:
            runs = range_dependency(prep["t_full"], prep["flux_full"], BLS_RANGES, n_periods=15000)
            row["bls"] = [{kk: r[kk] for kk in
                           ("period", "sde", "range", "edge_fraction", "at_edge", "n_transits_in_baseline")}
                          for r in runs]
            print("    BLS by search range:")
            print("      " + format_table(
                [[f"{r['range'][0]:g}-{r['range'][1]:g}", f"{r['period']:.2f}",
                  f"{r['sde']:.1f}", "YES" if r["at_edge"] else "no",
                  f"{r['n_transits_in_baseline']:.1f}"] for r in runs],
                ["range [d]", "P [d]", "SDE", "pinned to edge", "cycles in baseline"]
            ).replace("\n", "\n      "))

        print(f"    analytic (circular)  {row['analytic_circular']:8.2f} d")
        print(f"    analytic (b=0.70)    {row['analytic_b070']:8.2f} d")
        print(f"    analytic (e=0.30)    {row['analytic_e030']:8.2f} d")
        print(f"    CNN-PINN             {row['pinn']:8.2f} d   "
              f"[{row['pinn_lo']:.1f}, {row['pinn_hi']:.1f}] (1 sigma)")
        if t.true_period:
            err = abs(row["pinn"] - t.true_period) / t.true_period * 100
            inside = row["pinn_lo"] <= t.true_period <= row["pinn_hi"]
            print(f"    TRUE                 {t.true_period:8.2f} d   "
                  f"PINN error {err:.1f}%, true value {'inside' if inside else 'OUTSIDE'} 1 sigma")
        print()

        results.append(row)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(prep["t_full"], prep["flux_full"], ".", ms=1, alpha=0.5)
        axes[0].axvline(prep["t0"], color="r", lw=0.8)
        axes[0].set_xlabel("BTJD"); axes[0].set_ylabel("normalised flux")
        axes[0].set_title(f"{t.label} - detrended sector")
        axes[1].plot((prep["t_grid"] - prep["t0"]) * 24, prep["lc"], ".-", ms=3, lw=0.7)
        axes[1].set_xlabel("hours from mid-transit")
        axes[1].set_title(f"depth {prep['depth']*100:.3f}%  FWHM {prep['duration_hr']:.2f} h")
        fig.tight_layout()
        fig.savefig(B.FIGURES / f"real_{t.tic}.png", dpi=140)
        plt.close(fig)

    out = B.DATA / "real_data_results.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"[04] wrote {out}")


if __name__ == "__main__":
    main()
