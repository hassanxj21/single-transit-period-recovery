#!/usr/bin/env python
"""BLS vs analytic inversion vs CNN-PINN on single-transit targets, with BLS
given no information it would not have in practice.

The comparison has to be fair in the direction that matters: BLS is handed the
only search grid a real user could justify without already knowing the answer.
With one sector of data you cannot detect a period longer than the baseline, so
0.5-27 d is the honest default. A wider grid is run alongside to show that
widening it does not rescue the method - it just moves where the noise peaks.

    python scripts/06_head_to_head.py
    python scripts/06_head_to_head.py --tic 65910228
"""

import argparse
import json

import _bootstrap as B
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exopinn import data as D
from exopinn.bls import run_bls
from exopinn.constants import SECTOR_BASELINE_DAYS
from exopinn.evaluate import format_table
from exopinn.physics import equilibrium_temperature, period_from_duration
from exopinn.targets import CANDIDATES, CONFIRMED
from exopinn.train import load_model, predict

# The grid a user can justify with no prior knowledge, plus a deliberately
# generous one to show that generosity is not the missing ingredient.
GRID_HONEST = (0.5, SECTOR_BASELINE_DAYS)
GRID_WIDE = (0.5, 200.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int, default=None)
    ap.add_argument("--model", default=str(B.MODELS / "cnn_pinn" / "cnn_pinn.pt"))
    ap.add_argument("--n-periods", type=int, default=20000)
    ap.add_argument("--no-model", action="store_true",
                    help="run BLS + analytic only; use while the model is still training")
    args = ap.parse_args()

    model = norm = ckpt = None
    if not args.no_model:
        model, norm, ckpt = load_model(args.model)
        print(f"[06] model {args.model}")
        print(f"[06] trained with lambda_physics={ckpt['train_config']['lambda_physics']}, "
              f"{ckpt['train_config']['epochs']} epochs\n")
    else:
        print("[06] --no-model: BLS and analytic inversion only\n")

    targets = CONFIRMED + CANDIDATES
    if args.tic:
        targets = [t for t in targets if t.tic == args.tic] or targets

    rows, records = [], []
    for t in targets:
        print(f"--- {t.label} " + "-" * (60 - len(t.label)))
        try:
            prep = D.prepare_target(t.tic, sector=t.sector, search_window=t.window,
                                    mass=t.mass, radius=t.radius)
        except Exception as exc:
            print(f"    prepare failed: {type(exc).__name__}: {str(exc)[:100]}\n")
            continue
        if "error" in prep or "scalars" not in prep:
            print(f"    {prep.get('error', 'missing stellar properties')}\n")
            continue

        rec = {"tic": t.tic, "label": t.label, "true_period": t.true_period,
               "sector": prep["provenance"]["sector"], "author": prep["provenance"]["author"],
               "depth": prep["depth"], "duration_fwhm_hr": prep["duration_hr"],
               "snr": prep["snr"]}

        # --- BLS, no prior information -------------------------------------
        bls_h = run_bls(prep["t_full"], prep["flux_full"], *GRID_HONEST, n_periods=args.n_periods)
        bls_w = run_bls(prep["t_full"], prep["flux_full"], *GRID_WIDE, n_periods=args.n_periods)
        rec["bls_honest"] = {k: bls_h[k] for k in ("period", "sde", "at_edge", "n_transits_in_baseline")}
        rec["bls_wide"] = {k: bls_w[k] for k in ("period", "sde", "at_edge", "n_transits_in_baseline")}

        # --- analytic inversion --------------------------------------------
        k = np.sqrt(max(prep["depth"], 0.0))
        rec["analytic"] = period_from_duration(prep["duration_hr"], prep["mass"],
                                               prep["radius"], k=k, b=0.0, contact="fwhm")

        # --- CNN-PINN -------------------------------------------------------
        if model is not None:
            p, lo, hi = predict(model, norm, prep["lc"][None, :], prep["scalars"])
            rec["pinn"], rec["pinn_lo"], rec["pinn_hi"] = float(p[0]), float(lo[0]), float(hi[0])
        else:
            rec["pinn"] = rec["pinn_lo"] = rec["pinn_hi"] = float("nan")
        # cache the extracted features so the PINN column can be filled in later
        rec["lc_window"] = prep["lc"].tolist()
        rec["mass"], rec["radius"] = float(prep["mass"]), float(prep["radius"])

        teff = prep.get("teff") or t.teff
        rec["teff"] = teff
        if teff and np.isfinite(rec["pinn"]):
            rec["teq"] = float(equilibrium_temperature(teff, prep["radius"], rec["pinn"], prep["mass"]))

        print(f"    S{rec['sector']} {rec['author']}  depth={prep['depth']*100:.3f}%  "
              f"FWHM={prep['duration_hr']:.2f}h  SNR={prep['snr']:.1f}")
        print(f"    BLS  0.5-27 d : {bls_h['period']:8.2f} d  SDE={bls_h['sde']:5.1f}  "
              f"{'PINNED TO GRID EDGE' if bls_h['at_edge'] else ''}")
        print(f"    BLS  0.5-200 d: {bls_w['period']:8.2f} d  SDE={bls_w['sde']:5.1f}  "
              f"{'PINNED TO GRID EDGE' if bls_w['at_edge'] else ''}")
        print(f"    analytic      : {rec['analytic']:8.2f} d")
        if model is not None:
            print(f"    CNN-PINN      : {rec['pinn']:8.2f} d  [{rec['pinn_lo']:.1f}, {rec['pinn_hi']:.1f}]")

        def err(x):
            return "" if not t.true_period or not np.isfinite(x) else f"{abs(x - t.true_period) / t.true_period * 100:.0f}%"

        if t.true_period:
            inside = bool(rec["pinn_lo"] <= t.true_period <= rec["pinn_hi"])
            tail = "" if model is None else (f"  PINN {err(rec['pinn'])} "
                                             f"({'inside' if inside else 'OUTSIDE'} 1sigma)")
            print(f"    TRUE          : {t.true_period:8.2f} d   "
                  f"| BLS {err(bls_h['period'])}  analytic {err(rec['analytic'])}{tail}")
            rec["pinn_inside_1sigma"] = inside
        print()

        rows.append([t.label, f"{t.true_period:.2f}" if t.true_period else "unknown",
                     f"{bls_h['period']:.2f}", err(bls_h["period"]),
                     f"{rec['analytic']:.1f}", err(rec["analytic"]),
                     f"{rec['pinn']:.1f}", err(rec["pinn"]),
                     f"[{rec['pinn_lo']:.0f},{rec['pinn_hi']:.0f}]"])
        records.append(rec)

    print("=" * 100)
    print("HEAD TO HEAD - BLS given only what it could know in practice (0.5-27 d)")
    print("=" * 100)
    print(format_table(rows, ["target", "true P", "BLS", "err", "analytic", "err",
                              "CNN-PINN", "err", "PINN 1sigma"]))

    conf = [r for r in records if r.get("true_period")]
    if conf:
        print("\nOn the confirmed planets:")
        for key, name in [("bls_honest", "BLS (0.5-27 d)"), ("analytic", "analytic"), ("pinn", "CNN-PINN")]:
            vals = [(r[key]["period"] if isinstance(r[key], dict) else r[key], r["true_period"]) for r in conf]
            e = [abs(pp - tp) / tp * 100 for pp, tp in vals if np.isfinite(pp)]
            if not e:
                print(f"  {name:16s} (not available)")
                continue
            print(f"  {name:16s} median |error| = {np.median(e):7.1f}%   "
                  f"worst = {max(e):7.1f}%")
        n_in = sum(r.get("pinn_inside_1sigma", False) for r in conf)
        print(f"  CNN-PINN 1-sigma interval contained the truth for {n_in}/{len(conf)} confirmed planets")

    out = B.DATA / "head_to_head.json"
    out.write_text(json.dumps(records, indent=2, default=float))
    print(f"\n[06] wrote {out}")

    # figure
    if records:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        y = np.arange(len(records))
        for r, yy in zip(records, y):
            if r.get("true_period"):
                ax.plot(r["true_period"], yy, "k*", ms=16, zorder=5)
            ax.plot(r["bls_honest"]["period"], yy, "s", color="#b5442f", ms=8)
            ax.plot(r["analytic"], yy, "^", color="#3b6ea5", ms=8)
            ax.plot([r["pinn_lo"], r["pinn_hi"]], [yy, yy], "-", color="#2e8b57", lw=3, alpha=0.5)
            ax.plot(r["pinn"], yy, "o", color="#2e8b57", ms=9)
        ax.axvline(SECTOR_BASELINE_DAYS, ls="--", color="darkorange", label="27 d sector baseline")
        ax.plot([], [], "k*", ms=14, label="true period")
        ax.plot([], [], "s", color="#b5442f", label="BLS (0.5-27 d)")
        ax.plot([], [], "^", color="#3b6ea5", label="analytic inversion")
        ax.plot([], [], "o", color="#2e8b57", label="CNN-PINN (1 sigma)")
        ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in records])
        ax.set_xscale("log"); ax.set_xlabel("orbital period [d]")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="x", which="both")
        fig.tight_layout(); fig.savefig(B.FIGURES / "head_to_head.png", dpi=150)
        print(f"[06] figure -> {B.FIGURES / 'head_to_head.png'}")


if __name__ == "__main__":
    main()
