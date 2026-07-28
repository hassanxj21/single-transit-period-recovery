#!/usr/bin/env python
"""Definitive real-data comparison: every method, same targets, same features.

Methods
  BLS (0.5-27 d)      the only grid a user could justify without knowing the answer
  analytic            closed-form inversion, circular and central (b=0)
  synth full          CNN-PINN trained on fully simulated light curves
  synth scalars       same, CNN branch removed
  inj full            CNN-PINN trained on transits injected into REAL photometry
  inj scalars         same, CNN branch removed

The two "scalars" models are the controls. Comparing full-vs-scalars *within*
each training regime isolates the CNN branch; comparing synth-vs-inj isolates
the domain gap.

    python scripts/11_final_comparison.py
"""

import argparse
import json

import _bootstrap as B
import numpy as np

from exopinn import data as D
from exopinn.bls import run_bls
from exopinn.constants import SECTOR_BASELINE_DAYS
from exopinn.evaluate import format_table
from exopinn.physics import equilibrium_temperature, period_from_duration
from exopinn.targets import CANDIDATES, CONFIRMED
from exopinn.train import load_model, predict

MODELS = [
    ("synth full", "cnn_pinn_full"),
    ("synth scalars", "cnn_pinn_scalars_only"),
    ("inj full", "cnn_pinn_inj_full"),
    ("inj scalars", "cnn_pinn_inj_scalars"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bls", action="store_true")
    ap.add_argument("--n-periods", type=int, default=12000)
    args = ap.parse_args()

    models = {}
    for label, d in MODELS:
        path = B.MODELS / d / "cnn_pinn.pt"
        if path.exists():
            models[label] = load_model(path)
            print(f"[11] loaded {label:16s} <- {d}")
        else:
            print(f"[11] MISSING {label:16s} ({path})")
    print()

    records = []
    for t in CONFIRMED + CANDIDATES:
        try:
            prep = D.prepare_target(t.tic, sector=t.sector, search_window=t.window,
                                    mass=t.mass, radius=t.radius, verbose=False)
        except Exception as exc:
            print(f"  {t.label}: prepare failed ({type(exc).__name__})")
            continue
        if "error" in prep or "scalars" not in prep:
            print(f"  {t.label}: {prep.get('error', 'no stellar properties')}")
            continue

        rec = {"tic": t.tic, "label": t.label, "true_period": t.true_period,
               "depth": prep["depth"], "duration_hr": prep["duration_hr"],
               "snr": prep["snr"], "sector": prep["provenance"]["sector"]}

        if not args.no_bls:
            r = run_bls(prep["t_full"], prep["flux_full"], 0.5, SECTOR_BASELINE_DAYS,
                        n_periods=args.n_periods)
            rec["bls"] = float(r["period"])
            rec["bls_sde"] = float(r["sde"])

        k = np.sqrt(max(prep["depth"], 0.0))
        rec["analytic"] = float(period_from_duration(prep["duration_hr"], prep["mass"],
                                                     prep["radius"], k=k, b=0.0, contact="fwhm"))

        for label, _ in MODELS:
            if label not in models:
                continue
            m, n, _ = models[label]
            p, lo, hi = predict(m, n, prep["lc"][None, :], prep["scalars"])
            rec[label] = float(p[0])
            rec[label + " lo"] = float(lo[0])
            rec[label + " hi"] = float(hi[0])

        teff = prep.get("teff") or t.teff
        if teff and "inj scalars" in rec:
            rec["teq"] = float(equilibrium_temperature(teff, prep["radius"],
                                                       rec["inj scalars"], prep["mass"]))
        records.append(rec)

    method_names = (["bls", "analytic"] if not args.no_bls else ["analytic"]) + \
                   [lab for lab, _ in MODELS if lab in models]

    print("=" * 110)
    print("CONFIRMED PLANETS - predicted period [d] and error")
    print("=" * 110)
    conf = [r for r in records if r.get("true_period")]
    rows = []
    for r in conf:
        row = [r["label"], f"{r['true_period']:.2f}"]
        for mname in method_names:
            v = r.get(mname, np.nan)
            e = abs(v - r["true_period"]) / r["true_period"] * 100 if np.isfinite(v) else np.nan
            row.append(f"{v:.1f} ({e:.0f}%)" if np.isfinite(v) else "-")
        rows.append(row)
    print(format_table(rows, ["target", "true"] + method_names))

    print()
    print("=" * 110)
    print("SUMMARY on confirmed planets")
    print("=" * 110)
    rows = []
    for mname in method_names:
        errs, dex, cov, n_cov = [], [], 0, 0
        for r in conf:
            v = r.get(mname, np.nan)
            if not np.isfinite(v) or v <= 0:
                continue
            errs.append(abs(v - r["true_period"]) / r["true_period"] * 100)
            dex.append(abs(np.log10(v / r["true_period"])))
            if mname + " lo" in r:
                n_cov += 1
                cov += int(r[mname + " lo"] <= r["true_period"] <= r[mname + " hi"])
        if not errs:
            continue
        rows.append([mname, f"{np.median(errs):.1f}%", f"{max(errs):.0f}%",
                     f"{np.median(dex):.3f}",
                     f"{cov}/{n_cov}" if n_cov else "-"])
    print(format_table(rows, ["method", "median err", "worst", "median dex", "1sig cov"]))

    print()
    print("=" * 110)
    print("UNCONFIRMED CANDIDATES - period unknown, models disagree by how much?")
    print("=" * 110)
    rows = []
    for r in records:
        if r.get("true_period"):
            continue
        row = [r["label"], f"{r['duration_hr']:.1f}h", f"{r['depth']*100:.3f}%"]
        row += [f"{r.get(m, float('nan')):.1f}" for m in method_names]
        rows.append(row)
    print(format_table(rows, ["target", "FWHM", "depth"] + method_names))

    (B.DATA / "final_comparison.json").write_text(json.dumps(records, indent=2, default=float))
    print(f"\n[11] wrote {B.DATA / 'final_comparison.json'}")


if __name__ == "__main__":
    main()
