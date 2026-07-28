#!/usr/bin/env python
"""Full head-to-head on the expanded confirmed single-transit set.

Methods: BLS at two grids, analytic inversion, and every trained model.
Reports per-target prediction / error / interval / SDE / edge-pinning, then
medians, worst case and interval coverage per method, then breakdowns by
eccentricity and impact parameter using the SAME bins as the synthetic
analysis - so the claim "the CNN advantage decays as e rises" can be tested on
real data rather than asserted.

TIC 393818343 is excluded from every headline number: at P = 16.25 d it is not
a single-transit target. It is reported separately as physics validation.

    python scripts/13_expanded_headtohead.py
"""

import argparse
import json
import os
import time
import warnings

import _bootstrap as B
import numpy as np

from exopinn import data as D
from exopinn.bls import run_bls
from exopinn.constants import SECTOR_BASELINE_DAYS
from exopinn.evaluate import format_table
from exopinn.physics import period_from_duration
from exopinn.targets import CONFIRMED, EXPANDED, VALIDATION_ONLY
from exopinn.train import load_model, predict

warnings.filterwarnings("ignore")

MODEL_DIRS = [
    ("scalars", "cnn_pinn_scalars_only"),
    ("synth full", "cnn_pinn_full"),
    ("inj full", "cnn_pinn_inj_full"),
    ("inj scalars", "cnn_pinn_inj_scalars"),
]
GRID_A = (0.5, SECTOR_BASELINE_DAYS)
GRID_B = (0.5, 200.0)


def load_models():
    out = {}
    for label, d in MODEL_DIRS:
        p = B.MODELS / d / "cnn_pinn.pt"
        if p.exists():
            out[label] = load_model(p)
            print(f"[13] model {label:12s} <- {d}")
        else:
            print(f"[13] model {label:12s} MISSING ({d}) - column omitted")
    return out


def evaluate(t, models, n_periods, do_bls=True):
    sector, window = t.sector, t.window
    ephem = None
    if getattr(t, "tranmid", None) and t.true_period:
        hit = D.sector_with_transit(t.tic, t.tranmid, t.true_period)
        if hit is None:
            return {"tic": t.tic, "label": t.label,
                    "error": "no observed sector contains a predicted transit"}
        sector, window, ephem = hit["sector"], hit["window"], hit["t_transit"]
        n_cycles = hit["n_cycles"]
        sig_p = getattr(t, "period_err", None) or 0.0
        sig_t0 = getattr(t, "tranmid_err", None) or 0.0
        timing_unc = float(np.sqrt(sig_t0**2 + (n_cycles * sig_p) ** 2))

    p = D.prepare_target(t.tic, sector=sector, search_window=window,
                         mass=t.mass, radius=t.radius, verbose=False)
    if "error" in p or "scalars" not in p:
        return {"tic": t.tic, "label": t.label, "error": p.get("error", "no stellar props")}

    rec = {"tic": t.tic, "label": t.label, "true_period": t.true_period,
           "sector": p["provenance"]["sector"], "author": p["provenance"]["author"],
           "exptime": p["provenance"]["exptime"],
           "depth": p["depth"], "duration_hr": p["duration_hr"], "snr": p["snr"],
           "snr_point": p.get("snr_point"), "n_in_transit": p.get("n_in_transit"),
           "known_b": t.known_b, "known_ecc": t.known_ecc,
           "mass": p["mass"], "radius": p["radius"],
           "ephem_t0": ephem, "t0_measured": p["t0"],
           "n_cycles": locals().get("n_cycles"),
           "timing_unc_d": locals().get("timing_unc"),
           "published_dur": t.duration_hr}

    if do_bls:
        for tag, (lo, hi) in [("bls27", GRID_A), ("bls200", GRID_B)]:
            r = run_bls(p["t_full"], p["flux_full"], lo, hi, n_periods=n_periods)
            rec[tag] = float(r["period"])
            rec[tag + "_sde"] = float(r["sde"])
            rec[tag + "_edge"] = bool(r["at_edge"])
            rec[tag + "_cycles"] = float(r["n_transits_in_baseline"])

    k = np.sqrt(max(p["depth"], 0.0))
    rec["analytic"] = float(period_from_duration(p["duration_hr"], p["mass"], p["radius"],
                                                 k=k, b=0.0, contact="fwhm"))
    for label, (m, n, _) in models.items():
        pr, lo, hi = predict(m, n, p["lc"][None, :], p["scalars"])
        rec[label] = float(pr[0])
        rec[label + "_lo"] = float(lo[0])
        rec[label + "_hi"] = float(hi[0])
    return rec


def err(v, truth):
    return abs(v - truth) / truth * 100 if (np.isfinite(v) and truth) else np.nan


def summarise(records, methods, title):
    print(f"\n{'=' * 104}")
    print(title)
    print("=" * 104)
    rows = []
    for m in methods:
        e, cov, ncov = [], 0, 0
        for r in records:
            v = r.get(m, np.nan)
            if not np.isfinite(v) or v <= 0:
                continue
            e.append(err(v, r["true_period"]))
            if m + "_lo" in r:
                ncov += 1
                cov += int(r[m + "_lo"] <= r["true_period"] <= r[m + "_hi"])
        if not e:
            continue
        dex = [abs(np.log10(r.get(m) / r["true_period"])) for r in records
               if np.isfinite(r.get(m, np.nan)) and r.get(m, 0) > 0]
        rows.append([m, len(e), f"{np.median(e):.1f}%", f"{max(e):.0f}%",
                     f"{np.median(dex):.3f}",
                     f"{np.mean([x < 100 for x in e]):.0%}",
                     f"{cov}/{ncov} ({cov/ncov:.0%})" if ncov else "-"])
    print(format_table(rows, ["method", "n", "median err", "worst", "median dex",
                              "within 2x", "1sig coverage"]))


def breakdown(records, methods, key, bins, label):
    print(f"\n{'-' * 104}")
    print(f"BY {label.upper()} - median |error| (same bins as the synthetic analysis)")
    print("-" * 104)
    rows = []
    for lo, hi in bins:
        sel = [r for r in records
               if r.get(key) is not None and np.isfinite(r.get(key, np.nan))
               and lo <= r[key] < hi]
        if len(sel) < 2:
            continue
        row = [f"{lo:.2f}-{hi:.2f}", len(sel)]
        for m in methods:
            e = [err(r.get(m, np.nan), r["true_period"]) for r in sel
                 if np.isfinite(r.get(m, np.nan)) and r.get(m, 0) > 0]
            row.append(f"{np.median(e):.0f}%" if e else "-")
        rows.append(row)
    if rows:
        print(format_table(rows, [label, "n"] + methods))
    else:
        print(f"  too few targets with published {label} for a breakdown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-periods", type=int, default=12000)
    ap.add_argument("--no-bls", action="store_true")
    ap.add_argument("--min-snr", type=float, default=7.5,
                    help="targets whose extracted transit falls below this are reported "
                         "but excluded from headline numbers - you cannot measure a "
                         "duration from a transit you cannot detect")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--out", default=str(B.DATA / "expanded_results.json"))
    args = ap.parse_args()

    models = load_models()
    targets = CONFIRMED + EXPANDED
    print(f"\n[13] {len(targets)} confirmed single-transit targets "
          f"({len(CONFIRMED)} original + {len(EXPANDED)} expanded)")
    print(f"[13] {len(VALIDATION_ONLY)} validation-only target(s) reported separately\n")

    # Incremental results file: a crash at target 28 must not lose 27 targets.
    partial_path = args.out.replace(".json", "_partial.json")
    done = {}
    if args.resume and os.path.exists(partial_path):
        for r in json.loads(open(partial_path).read()):
            done[r.get("tic")] = r
        print(f"[13] resuming: {len(done)} targets already in {partial_path}\n")

    records, failed = [], []
    t_start = time.time()
    for i, t in enumerate(targets, 1):
        t_tgt = time.time()
        if t.tic in done and "error" not in done[t.tic]:
            rec = done[t.tic]
            print(f"  [{i:>2}/{len(targets)}] {t.label:20s} (cached)", flush=True)
        else:
            rec = evaluate(t, models, args.n_periods, do_bls=not args.no_bls)

        dt, tot = time.time() - t_tgt, time.time() - t_start
        if "error" in rec:
            failed.append((t.label, rec["error"]))
            print(f"  [{i:>2}/{len(targets)}] TIC {t.tic:<10} {t.label:18s} "
                  f"FAILED: {rec['error']}  [{dt:.0f}s, total {tot/60:.1f}m]", flush=True)
            continue
        rec["low_snr"] = bool(rec["snr"] < args.min_snr)
        records.append(rec)
        e_str = f"{err(rec.get('scalars', np.nan), t.true_period):.0f}%" \
            if "scalars" in rec else "-"
        print(f"  [{i:>2}/{len(targets)}] TIC {t.tic:<10} {t.label:18s} P={t.true_period:8.2f} "
              f"depth={rec['depth']*100:.3f}% FWHM={rec['duration_hr']:5.2f}h "
              f"SNR={rec['snr']:6.1f} err={e_str:>5}  "
              f"S{rec.get('sector')} n={rec.get('n_cycles')} "
              f"sig_t={rec.get('timing_unc_d') or 0:.4f}d  "
              f"[{dt:.0f}s, total {tot/60:.1f}m]", flush=True)
        with open(partial_path, "w") as fh:
            json.dump(records, fh, default=float)
        # Human-readable incremental CSV as well: a crash must not cost the run.
        import csv as _csv
        csv_path = partial_path.replace(".json", ".csv")
        keys = ["tic", "label", "true_period", "sector", "n_cycles", "timing_unc_d",
                "depth", "duration_hr", "snr", "known_b", "known_ecc",
                "bls27", "bls27_sde", "bls27_edge", "bls200", "bls200_sde",
                "analytic", "scalars", "synth full", "inj full", "inj scalars"]
        with open(csv_path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(keys)
            for rr in records:
                w.writerow([rr.get(k) for k in keys])

    val = []
    for t in VALIDATION_ONLY:
        r = evaluate(t, models, args.n_periods, do_bls=not args.no_bls)
        if "error" not in r:
            val.append(r)

    methods = (["bls27", "bls200"] if not args.no_bls else []) + ["analytic"] + list(models)
    good = [r for r in records if not r.get("low_snr")]
    weak = [r for r in records if r.get("low_snr")]
    print(f"\n[13] {len(good)} targets above SNR {args.min_snr}; "
          f"{len(weak)} below and excluded from headline numbers")
    for r in weak:
        print(f"       excluded: {r['label']:20s} SNR={r['snr']:.1f} "
              f"depth={r['depth']*100:.3f}% FWHM={r['duration_hr']:.2f}h "
              f"(published T14 {r.get('published_dur') or float('nan'):.2f}h)")

    # ---------------- per-target table ----------------
    print(f"\n{'=' * 104}")
    print("PER-TARGET RESULTS (expanded confirmed single-transit set)")
    print("=" * 104)
    rows = []
    for r in sorted(records, key=lambda x: x["true_period"]):
        row = [r["label"], f"{r['true_period']:.1f}",
               f"{r['known_ecc']:.2f}" if r.get("known_ecc") is not None else "-",
               f"{r['known_b']:.2f}" if r.get("known_b") is not None else "-"]
        for m in methods:
            v = r.get(m, np.nan)
            row.append(f"{v:.1f} ({err(v, r['true_period']):.0f}%)" if np.isfinite(v) else "-")
        rows.append(row)
    print(format_table(rows, ["target", "true P", "e", "b"] + methods))

    if not args.no_bls:
        print(f"\n{'-' * 104}")
        print("BLS DIAGNOSTICS")
        print("-" * 104)
        rows = [[r["label"], f"{r['bls27']:.2f}", f"{r['bls27_sde']:.1f}",
                 "YES" if r["bls27_edge"] else "no", f"{r['bls27_cycles']:.2f}",
                 f"{r['bls200']:.2f}", f"{r['bls200_sde']:.1f}",
                 "YES" if r["bls200_edge"] else "no"]
                for r in sorted(records, key=lambda x: x["true_period"])]
        print(format_table(rows, ["target", "P(0.5-27)", "SDE", "edge?", "cycles in baseline",
                                  "P(0.5-200)", "SDE", "edge?"]))

    summarise(good, methods, f"SUMMARY - detectable single-transit targets, n={len(good)}")
    if weak:
        summarise(records, methods, f"SUMMARY - all processed incl. low-SNR, n={len(records)}")

    ebins = [(0.0, 1e-9), (1e-9, 0.15), (0.15, 0.35), (0.35, 0.95)]
    bbins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.2)]
    model_methods = ["analytic"] + list(models)
    breakdown(good, model_methods, "known_ecc", ebins, "eccentricity")
    breakdown(good, model_methods, "known_b", bbins, "impact parameter")

    if val:
        print(f"\n{'=' * 104}")
        print("VALIDATION-ONLY (NOT single-transit; excluded from all numbers above)")
        print("=" * 104)
        rows = []
        for r in val:
            row = [r["label"], f"{r['true_period']:.2f}"]
            for m in methods:
                v = r.get(m, np.nan)
                row.append(f"{v:.1f} ({err(v, r['true_period']):.0f}%)" if np.isfinite(v) else "-")
            rows.append(row)
        print(format_table(rows, ["target", "true P"] + methods))

    if failed:
        print(f"\n{len(failed)} targets failed to process:")
        for name, why in failed:
            print(f"    {name:24s} {why}")

    payload = {"records": records, "good": [r["tic"] for r in good],
               "min_snr": args.min_snr, "validation_only": val,
               "failed": failed, "methods": methods}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\n[13] wrote {args.out}")


if __name__ == "__main__":
    main()
