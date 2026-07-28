#!/usr/bin/env python
"""Produce RESULTS_DUMP.txt and the figure set from the expanded head-to-head.

    python scripts/14_results_dump.py
"""

import argparse
import json
import warnings

import _bootstrap as B
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exopinn.evaluate import format_table

warnings.filterwarnings("ignore")

EBINS = [(0.0, 1e-9, "e = 0"), (1e-9, 0.15, "0 < e < 0.15"),
         (0.15, 0.35, "0.15-0.35"), (0.35, 0.95, "e >= 0.35")]
BBINS = [(0.0, 0.3, "b < 0.3"), (0.3, 0.5, "0.3-0.5"),
         (0.5, 0.7, "0.5-0.7"), (0.7, 1.2, "b >= 0.7")]
COLORS = {"bls27": "#b5442f", "bls200": "#d98b7a", "analytic": "#3b6ea5",
          "scalars": "#c08a3e", "synth full": "#2e8b57", "inj full": "#7b4fa8",
          "inj scalars": "#a88fd0"}


def err(v, truth):
    return abs(v - truth) / truth * 100 if (np.isfinite(v) and truth) else np.nan


def stat_block(records, methods, out):
    rows = []
    for m in methods:
        e = [err(r.get(m, np.nan), r["true_period"]) for r in records
             if np.isfinite(r.get(m, np.nan)) and r.get(m, 0) > 0]
        if not e:
            continue
        dex = [abs(np.log10(r[m] / r["true_period"])) for r in records
               if np.isfinite(r.get(m, np.nan)) and r.get(m, 0) > 0]
        cov = ncov = 0
        for r in records:
            if m + "_lo" in r and np.isfinite(r.get(m, np.nan)):
                ncov += 1
                cov += int(r[m + "_lo"] <= r["true_period"] <= r[m + "_hi"])
        rows.append([m, len(e), f"{np.median(e):.1f}%", f"{np.mean(e):.1f}%", f"{max(e):.0f}%",
                     f"{np.median(dex):.3f}", f"{np.mean([x < 100 for x in e]):.0%}",
                     f"{cov}/{ncov} ({cov/ncov:.0%})" if ncov else "-"])
    out.append(format_table(rows, ["method", "n", "median err", "mean err", "worst",
                                   "median dex", "within 2x", "1sig coverage"]))


def bin_block(records, methods, key, bins, label, out):
    out.append(f"\nBY {label.upper()} - median |error| %")
    rows = []
    for lo, hi, name in bins:
        sel = [r for r in records if r.get(key) is not None
               and np.isfinite(r.get(key, np.nan)) and lo <= r[key] < hi]
        if len(sel) < 2:
            rows.append([name, len(sel)] + ["-"] * len(methods))
            continue
        row = [name, len(sel)]
        for m in methods:
            e = [err(r.get(m, np.nan), r["true_period"]) for r in sel
                 if np.isfinite(r.get(m, np.nan)) and r.get(m, 0) > 0]
            row.append(f"{np.median(e):.0f}" if e else "-")
        rows.append(row)
    out.append(format_table(rows, [label, "n"] + methods))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(B.DATA / "expanded_results.json"))
    args = ap.parse_args()

    payload = json.loads(open(args.results).read())
    records = payload["records"]
    val = payload.get("validation_only", [])
    methods = payload["methods"]
    model_methods = [m for m in methods if m not in ("bls27", "bls200")]

    # ---- exclusion arithmetic -------------------------------------------
    PRIOR_P_MAX, PRIOR_P_MIN = 1000.0, 3.0
    min_snr = payload.get("min_snr", 5.0)

    excl_snr = [r for r in records if r["snr"] < min_snr]
    rest = [r for r in records if r["snr"] >= min_snr]
    excl_prior = [r for r in rest
                  if r["true_period"] > PRIOR_P_MAX or r["true_period"] < PRIOR_P_MIN]
    headline = [r for r in rest if r not in excl_prior]
    all_processed = list(records)
    high_b = [r for r in headline
              if r.get("known_b") is not None and r["known_b"] > 1.0]

    out = []
    A = out.append
    A("=" * 108)
    A("EXOPLANET SINGLE-TRANSIT PERIOD RECOVERY - RESULTS DUMP")
    A("=" * 108)

    A("\n" + "=" * 108)
    A("0. SAMPLE ARITHMETIC - what the headline n actually is")
    A("=" * 108)
    A(f"  N  selected confirmed single-transit targets      {len(records) + len(payload.get('failed', []))}")
    A(f"     of which failed to process                     {len(payload.get('failed', []))}")
    A(f"     processed successfully                         {len(records)}")
    A(f"  M  excluded, transit SNR < {min_snr:<28.0f} {len(excl_snr)}")
    A(f"  K  excluded, true period outside training prior   {len(excl_prior)}")
    A(f"     ({PRIOR_P_MIN:.0f} - {PRIOR_P_MAX:.0f} d)")
    A(f"  n  HEADLINE SET                                   {len(headline)}")
    A("")
    A("  One-sentence form for the paper:")
    A(f'    "Of {len(records) + len(payload.get("failed", []))} confirmed single-transit planets selected, '
      f'{len(payload.get("failed", []))} failed to process,')
    A(f'     {len(excl_snr)} were excluded because the transit was not detected above SNR {min_snr:.0f},')
    A(f'     and {len(excl_prior)} because the true period lies outside the model\'s training')
    A(f'     prior, leaving n = {len(headline)}."')

    if excl_snr:
        A(f"\n  M - EXCLUDED ON SNR (< {min_snr:.0f}):")
        rows = [[r["label"], f"{r['true_period']:.1f}", f"{r['snr']:.1f}",
                 f"{r['depth']*100:.3f}", f"{r['duration_hr']:.2f}",
                 f"{r.get('published_dur') or float('nan'):.2f}"]
                for r in sorted(excl_snr, key=lambda x: x["snr"])]
        A(format_table(rows, ["target", "true P", "SNR", "depth%", "FWHM meas", "T14 pub"]))
        A("  These are genuine non-detections in a single sector, not pipeline failures:")
        A("  a duration cannot be measured from a transit that is not detected.")

    if excl_prior:
        A("\n  K - EXCLUDED AS PRIOR-LIMITED (supplementary table):")
        A("  Prior saturation measured rather than asserted - each row shows what the")
        A("  model returned, the prior edge it saturated against, and the truth.")
        rows = []
        for r in sorted(excl_prior, key=lambda x: x["true_period"]):
            edge = PRIOR_P_MAX if r["true_period"] > PRIOR_P_MAX else PRIOR_P_MIN
            row = [r["label"], f"{r['true_period']:.1f}", f"{edge:.0f}"]
            for m in model_methods:
                v = r.get(m, np.nan)
                row.append(f"{v:.1f}" if np.isfinite(v) else "-")
            row.append(f"{r['snr']:.1f}")
            rows.append(row)
        A(format_table(rows, ["target", "true P", "prior edge"] + model_methods + ["SNR"]))

    if high_b:
        A("\n  GEOMETRIC REGIME FLAG - targets with published b > 1.0:")
        A("  Above b = 1 the planet centre never crosses the stellar disc; only part of")
        A("  the planet ever overlaps. The duration relation still holds while b < 1+k,")
        A("  but this is a distinct regime from ordinary grazing and is sparsely covered")
        A("  by training (b is drawn as U(0, 1.05) x (1+k), so b > 1 appears only for")
        A("  larger radius ratios). These rows are retained in the headline set but")
        A("  flagged; a reader should not treat them as ordinary transits.")
        rows = []
        for r in high_b:
            row = [r["label"], f"{r['true_period']:.1f}", f"{r['known_b']:.3f}",
                   f"{np.sqrt(max(r['depth'], 0)):.3f}"]
            for m in model_methods:
                v = r.get(m, np.nan)
                row.append(f"{v:.1f} ({err(v, r['true_period']):.0f}%)" if np.isfinite(v) else "-")
            rows.append(row)
        A(format_table(rows, ["target", "true P", "published b", "k(meas)"] + model_methods))

    A(f"\nMethods: {', '.join(methods)}")
    A("\nTIC 393818343 is EXCLUDED from all numbers below: at P = 16.25 d against a")
    A("25.4 d usable sector baseline it completes ~1.6 orbits per sector and is not")
    A("a single-transit target. It is reported separately as physics validation.")

    records = headline  # every table below is the headline set

    A("\n" + "=" * 108)
    A("1. PER-TARGET RESULTS")
    A("=" * 108)
    rows = []
    for r in sorted(records, key=lambda x: x["true_period"]):
        row = [r["label"], f"{r['true_period']:.1f}",
               f"{r['known_ecc']:.2f}" if r.get("known_ecc") is not None else "-",
               f"{r['known_b']:.2f}" if r.get("known_b") is not None else "-",
               f"{r['depth']*100:.3f}", f"{r['duration_hr']:.2f}", f"{r['snr']:.1f}"]
        for m in methods:
            v = r.get(m, np.nan)
            row.append(f"{v:.1f} ({err(v, r['true_period']):.0f}%)" if np.isfinite(v) else "-")
        rows.append(row)
    A(format_table(rows, ["target", "true P", "e", "b", "depth%", "FWHM", "SNR"] + methods))

    A("\n" + "=" * 108)
    A("2. SUMMARY - all confirmed single-transit targets")
    A("=" * 108)
    stat_block(records, methods, out)

    A("\n" + "=" * 108)
    A("3. IMPACT PARAMETER  ** PRIMARY RESULT ** - continuous, unbinned")
    A("=" * 108)
    A("Reported as a rank correlation rather than binned means: with n of this size,")
    A("four bins give 4-5 targets each and the result becomes a comparison of noisy")
    A("means plus bin edges that have to be defended. The Spearman coefficient uses")
    A("every target and makes no arbitrary choices. Binned table follows for reference.")
    A("")
    A("  Quantity: gain = (analytic |error| - model |error|) in percentage points,")
    A("  against published impact parameter b. The analytic estimator must assume")
    A("  b = 0, so if a model reads b from transit shape, gain should RISE with b")
    A("  (positive rho). If the model only reproduces the analytic answer plus a")
    A("  prior, gain should be uncorrelated with b (rho ~ 0).")
    A("")
    try:
        from scipy.stats import spearmanr
        rows = []
        for m in [x for x in model_methods if x != "analytic"]:
            bs, gains = [], []
            for r in records:
                b = r.get("known_b")
                va, vm = r.get("analytic", np.nan), r.get(m, np.nan)
                if b is None or not np.isfinite(b):
                    continue
                if not (np.isfinite(va) and va > 0 and np.isfinite(vm) and vm > 0):
                    continue
                bs.append(b)
                gains.append(err(va, r["true_period"]) - err(vm, r["true_period"]))
            if len(bs) < 5:
                rows.append([m, len(bs), "-", "-", "-", "too few"])
                continue
            rho, pval = spearmanr(bs, gains)
            rows.append([m, len(bs), f"{np.median(gains):+.1f} pp", f"{rho:+.3f}",
                         f"{pval:.3f}",
                         "supports shape-reading" if (rho > 0 and pval < 0.05)
                         else "not significant"])
        A(format_table(rows, ["model", "n", "median gain", "Spearman rho",
                              "p-value", "verdict"]))
        A("")
        A("  Per-target values (b, gain in pp) for the scatter in fig04:")
        for m in [x for x in model_methods if x != "analytic"][:1]:
            pts = []
            for r in sorted(records, key=lambda x: (x.get("known_b") or 0)):
                b = r.get("known_b")
                va, vm = r.get("analytic", np.nan), r.get(m, np.nan)
                if b is None or not (np.isfinite(va) and np.isfinite(vm)):
                    continue
                pts.append([r["label"], f"{b:.3f}",
                            f"{err(va, r['true_period']):.0f}%",
                            f"{err(vm, r['true_period']):.0f}%",
                            f"{err(va, r['true_period']) - err(vm, r['true_period']):+.0f}"])
            A(format_table(pts, ["target", "b", "analytic err", f"{m} err", "gain pp"]))
    except Exception as exc:
        A(f"  Spearman analysis unavailable: {type(exc).__name__}: {exc}")

    A("\n" + "-" * 108)
    A("3a. BINNED BY IMPACT PARAMETER (reference only - see rank correlation above)")
    A("-" * 108)
    A("This is the rung of the argument that is otherwise unproven: what does the")
    A("network add over the analytic inversion? The analytic estimator must assume")
    A("b = 0, so its error should grow with true b. If a model reads b from transit")
    A("shape, its error should grow more slowly. If it does not, the model is only")
    A("reproducing the analytic answer plus a prior.")
    bin_block(records, model_methods, "known_b", BBINS, "impact parameter", out)

    A("\n  analytic-minus-model error by b bin (positive = model beats analytic):")
    rows = []
    for lo, hi, name in BBINS:
        sel = [r for r in records if r.get("known_b") is not None
               and np.isfinite(r.get("known_b", np.nan)) and lo <= r["known_b"] < hi]
        if len(sel) < 2:
            continue
        ea = [err(r.get("analytic", np.nan), r["true_period"]) for r in sel
              if np.isfinite(r.get("analytic", np.nan))]
        row = [name, len(sel), f"{np.median(ea):.0f}%" if ea else "-"]
        for m in [x for x in model_methods if x != "analytic"]:
            em = [err(r.get(m, np.nan), r["true_period"]) for r in sel
                  if np.isfinite(r.get(m, np.nan)) and r.get(m, 0) > 0]
            row.append(f"{np.median(ea) - np.median(em):+.0f} pp" if (em and ea) else "-")
        rows.append(row)
    if rows:
        A(format_table(rows, ["b bin", "n", "analytic"] +
                       [f"{m} gain" for m in model_methods if m != "analytic"]))

    A("\n" + "=" * 108)
    A("3b. BREAKDOWN BY ECCENTRICITY")
    A("=" * 108)
    A("The synthetic analysis found the CNN advantage decaying to zero as e rises,")
    A("because the CNN reads impact parameter from transit shape while eccentricity")
    A("is unobservable from a single transit. These bins test that on real data.")
    bin_block(records, model_methods, "known_ecc", EBINS, "eccentricity", out)

    A("\n" + "-" * 108)
    A("3b-sens. SENSITIVITY OF THE RANK CORRELATION TO THE SNR GATE")
    A("-" * 108)
    A("The gate at 7.5 comes from an approximation (sigma_T/T ~ 0.45/Q), so any")
    A("target sitting close to it makes the threshold a judgement call. This checks")
    A("whether including borderline targets changes the sign or the significance of")
    A("the primary result. If it does not, the threshold choice is immaterial and")
    A("the paper can say so.")
    try:
        from scipy.stats import spearmanr as _sp

        border = [r for r in all_processed
                  if r["snr"] < min_snr and r["snr"] >= 0.85 * min_snr]
        border_desc = ", ".join(
            "{} (SNR {:.1f})".format(r["label"], r["snr"]) for r in border) or "none"
        A(f"\n  borderline targets (0.85 x gate <= SNR < gate = "
          f"{0.85 * min_snr:.2f}-{min_snr:.2f}): {border_desc}")

        def rho_for(recs, m):
            bs, gs = [], []
            for r in recs:
                b = r.get("known_b")
                va, vm = r.get("analytic", np.nan), r.get(m, np.nan)
                if b is None or not np.isfinite(b):
                    continue
                if not (np.isfinite(va) and va > 0 and np.isfinite(vm) and vm > 0):
                    continue
                bs.append(b)
                gs.append(err(va, r["true_period"]) - err(vm, r["true_period"]))
            if len(bs) < 5:
                return None
            rho, pv = _sp(bs, gs)
            return len(bs), rho, pv

        rows = []
        for m in [x for x in model_methods if x != "analytic"]:
            base = rho_for(records, m)
            withb = rho_for(records + border, m)
            if base is None:
                continue
            if withb is None:
                rows.append([m, f"{base[1]:+.3f}", f"{base[2]:.3f}", "-", "-", "n/a"])
                continue
            sign_flip = np.sign(base[1]) != np.sign(withb[1])
            sig_flip = (base[2] < 0.05) != (withb[2] < 0.05)
            rows.append([m, f"{base[1]:+.3f} (n={base[0]})", f"{base[2]:.3f}",
                         f"{withb[1]:+.3f} (n={withb[0]})", f"{withb[2]:.3f}",
                         "CHANGES" if (sign_flip or sig_flip) else "unchanged"])
        if rows:
            A(format_table(rows, ["model", "rho (gated)", "p", "rho (+borderline)",
                                  "p", "sign/significance"]))
            A("")
            if all(r[-1] == "unchanged" for r in rows):
                A("  VERDICT: no model changes sign or significance when the borderline")
                A("  targets are included. The 7.5 threshold is immaterial to the primary")
                A("  result and the paper can state that explicitly.")
            else:
                A("  VERDICT: at least one model changes sign or significance. The")
                A("  threshold is NOT immaterial and must be justified, not asserted.")
    except Exception as exc:
        A(f"  sensitivity check unavailable: {type(exc).__name__}: {exc}")

    A("\n" + "=" * 108)
    A("3c. DID THE DOMAIN GAP CLOSE?  ** PRIMARY RESULT **")
    A("=" * 108)
    A("Out-of-distribution by construction: the injection noise pool excludes every")
    A("evaluation star, so these targets are stars the pool has never seen. Unlike")
    A("the in-distribution 44%/66% split, this does test transfer.")
    A("")
    med = {}
    for m in model_methods:
        e = [err(r.get(m, np.nan), r["true_period"]) for r in records
             if np.isfinite(r.get(m, np.nan)) and r.get(m, 0) > 0]
        if e:
            med[m] = float(np.median(e))
    rows = [[m, f"{v:.1f}%"] for m, v in med.items()]
    A(format_table(rows, ["method", "median |error| on real targets"]))
    A("")
    if "inj full" in med and "inj scalars" in med:
        f_, s_ = med["inj full"], med["inj scalars"]
        A(f"  inj_full {f_:.1f}%  vs  inj_scalars {s_:.1f}%   "
          f"(difference {s_ - f_:+.1f} pp)")
        if f_ < s_ * 0.95:
            A("  ONE-LINE VERDICT: the CNN branch beats its scalars-only control on real,")
            A("  unseen stars - consistent with injection-recovery having closed the gap.")
        elif f_ > s_ * 1.05:
            A("  ONE-LINE VERDICT: the CNN branch still LOSES to its scalars-only control")
            A("  on real stars - injection-recovery did NOT close the domain gap.")
        else:
            A("  ONE-LINE VERDICT: the CNN branch and its scalars-only control are")
            A("  indistinguishable on real stars - no evidence the CNN adds anything.")
    if "synth full" in med and "scalars" in med:
        A(f"\n  For comparison, synthetic-trained: full {med['synth full']:.1f}% vs "
          f"scalars {med['scalars']:.1f}%")

    A("\n" + "=" * 108)
    A("3d. EPHEMERIS EXTRAPOLATION ACROSS THE SAMPLE")
    A("=" * 108)
    A("Sector choice matters more than expected. TOI-2134 c measured at n=0 (S52)")
    A("gives depth 1.008% and SNR 182; the same ephemeris extrapolated to n=8 (S80)")
    A("gives 0.053% and SNR 2.5 - a factor of 19 in depth from sector choice alone.")
    A("If many targets sit at high n, ephemeris extrapolation is a systematic across")
    A("the sample rather than a one-off, and the paper must say so.")
    ncyc = [r.get("n_cycles") for r in records if r.get("n_cycles") is not None]
    if ncyc:
        ncyc = np.array(ncyc, float)
        A(f"\n  n = 0 (measured epoch)      {int((ncyc == 0).sum())}/{len(ncyc)}")
        A(f"  1 <= n <= 3                {int(((ncyc >= 1) & (ncyc <= 3)).sum())}/{len(ncyc)}")
        A(f"  n > 3 (far extrapolation)  {int((ncyc > 3).sum())}/{len(ncyc)}")
        A(f"  median n = {np.median(ncyc):.0f}, max n = {ncyc.max():.0f}")
        unc = [r.get("timing_unc_d") for r in records if r.get("timing_unc_d")]
        if unc:
            A(f"  propagated timing uncertainty: median {np.median(unc)*24:.2f} h, "
              f"max {max(unc)*24:.2f} h")
            A(f"  (search half-window is 0.75 d = 18 h, so targets whose propagated")
            A(f"   uncertainty approaches that are at risk of a missed epoch)")

    A("\n" + "=" * 108)
    A("4. BLS DIAGNOSTICS")
    A("=" * 108)
    if "bls27" in methods:
        rows = []
        n_edge = 0
        for r in sorted(records, key=lambda x: x["true_period"]):
            rows.append([r["label"], f"{r['true_period']:.1f}", f"{r['bls27']:.2f}",
                         f"{r['bls27_sde']:.1f}", "YES" if r["bls27_edge"] else "no",
                         f"{r['bls27_cycles']:.2f}", f"{r['bls200']:.2f}",
                         f"{r['bls200_sde']:.1f}", "YES" if r["bls200_edge"] else "no"])
            n_edge += int(r["bls27_edge"])
        A(format_table(rows, ["target", "true P", "P(0.5-27)", "SDE", "edge?",
                              "cycles", "P(0.5-200)", "SDE", "edge?"]))
        e27 = [err(r["bls27"], r["true_period"]) for r in records]
        e200 = [err(r["bls200"], r["true_period"]) for r in records]
        A(f"\n  median error, 0.5-27 d grid : {np.median(e27):.1f}%")
        A(f"  median error, 0.5-200 d grid: {np.median(e200):.1f}%")
        A(f"  widening the grid changes median error by {np.median(e200)-np.median(e27):+.1f} pp")
        A(f"  targets where BLS pinned to a grid edge: {n_edge}/{len(records)}")
        A(f"  median BLS SDE: {np.median([r['bls27_sde'] for r in records]):.2f} "
          f"(a real detection is conventionally SDE > 7)")

    A("\n" + "=" * 108)
    A("5. VALIDATION-ONLY TARGET (not single-transit, excluded above)")
    A("=" * 108)
    if val:
        rows = []
        for r in val:
            row = [r["label"], f"{r['true_period']:.2f}"]
            for m in methods:
                v = r.get(m, np.nan)
                row.append(f"{v:.1f} ({err(v, r['true_period']):.0f}%)" if np.isfinite(v) else "-")
            rows.append(row)
        A(format_table(rows, ["target", "true P"] + methods))
    else:
        A("  none processed")

    if payload.get("failed"):
        A("\n" + "=" * 108)
        A("6. FAILED TARGETS")
        A("=" * 108)
        for name, why in payload["failed"]:
            A(f"  {name:26s} {why}")

    A("\n" + "=" * 108)
    A("7. PIPELINE DEFECTS FOUND AND FIXED (each changed published numbers)")
    A("=" * 108)
    for i, (what, effect) in enumerate([
        ("lightkurve flatten() mask polarity inverted",
         "mask=True means EXCLUDE from the fit; passing ~mask fitted the trend "
         "through the transit. TOI-1899 depth 0.926% -> 2.911%."),
        ("transit mask wider than the detrend window",
         "masked +/-1.5 d while filtering on a 1 d window, so near the transit the "
         "filter had no unmasked data and its interpolated trend sagged into the dip."),
        ("first detrend pass destroyed long transits",
         "at a 1 d window NGTS-38 b's 11.4 h transit measured 3.05 h / 0.075%; that "
         "bad duration then sized the second-pass mask. First pass is now 5 d and "
         "the mask width iterates to convergence."),
        ("missing iterative sigma-clipping in the detrender",
         "lightkurve clips points below the trend, which incidentally protects "
         "transits. Without it NGTS-38 b read 0.118% against a published 0.349%."),
        ("filter window unbounded relative to segment length",
         "detrend_days = 8 x duration gives a 14 d filter for a 40 h transit, whose "
         "Savitzky-Golay edge effects reach past the centre of a short segment but "
         "not of a full sector - a length-dependent distortion. Now capped at 0.30 x span."),
        ("sector chosen as 'first available SPOC product'",
         "without checking whether a transit occurs in it. For long-period planets "
         "most sectors contain none, so the pipeline measured noise: HD 56414 b read "
         "1.82 h at SNR 1.6 against a published 7.58 h; TOI-201 c read 1.77 h at SNR "
         "1.2. Fixed by resolving the sector from the published ephemeris. Effect on "
         "TOI-2134 c: SNR 7.9 -> 60.3, depth 0.142% -> 1.008%."),
        ("context_days removed from InjectionConfig while generate() still used it",
         "crashed the injection build mid-edit; caught only because the log was "
         "being watched."),
        ("synthetic periods below the window length",
         "put a second transit in the 2.5 d array, letting the network read period "
         "off transit spacing rather than duration. Period floor raised to 3 d."),
        ("per-cadence SNR cut admitted invisible transits",
         "whose measured duration was set by noise. Replaced with integrated "
         "depth/sigma x sqrt(N_in) > 10."),
        ("evaluation targets present in the injection noise pool",
         "all 8 were donating their systematics to training. Pool rebuilt from 40 "
         "field stars with every target hard-excluded."),
        ("Gaussian NLL drove sigma to zero",
         "train NLL fell to -1.42 while val NLL blew up to +6.07 over 300 epochs; "
         "best epoch was 15. Fixed with a sigma floor, early stopping and "
         "per-improvement checkpointing."),
        ("sector selected without checking a transit occurs in it",
         "for long-period planets most observed sectors contain no transit, so the "
         "pipeline measured noise: HD 56414 b read 1.82 h at SNR 1.6 against a "
         "published 7.58 h; TOI-201 c read 1.77 h at SNR 1.2. Fixed by resolving "
         "the sector from the published ephemeris."),
        ("ephemeris merged on tic_id alone",
         "multi-planet systems share a TIC, so five targets carried another "
         "planet's transit time: TOI-201 c, TOI-2295 b, HD 28109 d, TOI-904 c and "
         "HD 22946 d. Fixed by asserting pl_name matches exactly, enforced as a "
         "pre-flight gate that refuses to launch on any mismatch."),
        ("sector candidates ranked by transit centrality, not cycle count",
         "ephemeris timing uncertainty compounds with the number of cycles "
         "extrapolated from the measured epoch, sigma_t(n) = sqrt(sigma_T0^2 + "
         "(n sigma_P)^2). Ranking by how centrally the transit sat in its sector "
         "ignored this: TOI-2134 c selected S80 (n=8) over S52 (n=0), giving depth "
         "0.053% vs 1.008% and SNR 2.5 vs 182 - a factor of 19 in depth from sector "
         "choice alone. Fixed by minimising |n|, with an SNR-validated fallback."),
    ], 1):
        A(f"  {i:2d}. {what}")
        A(f"      {effect}")

    A("\n" + "=" * 108)
    A("8. CLAIMS NOT YET SUPPORTED")
    A("=" * 108)
    A("  - Injection-recovery closing the sim-to-real gap. inj_full 44% vs inj_scalars")
    A("    66% is IN-DISTRIBUTION: measured on held-out data from the same 40-star pool")
    A("    it trained on, sharing systematics with training. This is identical in")
    A("    character to the synthetic result (38.2% vs 57.3%) that then failed to")
    A("    transfer to real targets. Only the real-target columns above answer whether")
    A("    the gap closed.")
    A("  - Any claim resting on the original n = 3. TIC 393818343 is not single-transit,")
    A("    so the pre-expansion sample was n = 2.")

    text = "\n".join(out)
    path = B.RESULTS / "RESULTS_DUMP.txt"
    path.write_text(text)
    print(text)
    print(f"\n[14] wrote {path}")

    # ------------------------------------------------------------------ figures
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    written_figs = []

    # fig01 predicted vs true
    fig, ax = plt.subplots(figsize=(7, 6.4))
    truths = np.array([r["true_period"] for r in records])
    lim = [truths.min() * 0.4, truths.max() * 2.5]
    for m in methods:
        v = np.array([r.get(m, np.nan) for r in records])
        ok = np.isfinite(v) & (v > 0)
        ax.scatter(truths[ok], v[ok], s=34, alpha=0.8, label=m,
                   color=COLORS.get(m, "gray"), edgecolor="k", linewidth=0.3)
    ax.plot(lim, lim, "k-", lw=1)
    for f in (2.0, 0.5):
        ax.plot(lim, [f * x for x in lim], "--", color="gray", lw=0.7)
    ax.axhspan(lim[0], 27, color="orange", alpha=0.07)
    ax.text(lim[0] * 1.2, 20, "BLS reachable\n(<27 d)", fontsize=7, color="darkorange")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true period [d]"); ax.set_ylabel("predicted period [d]")
    ax.set_title(f"Single-transit period recovery, n={len(records)} confirmed planets")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(B.FIGURES / "fig01_predicted_vs_true.png"); written_figs.append("fig01_predicted_vs_true.png")
    plt.close(fig)

    # fig02 median error bars
    fig, ax = plt.subplots(figsize=(7, 4.2))
    meds, labs, cols = [], [], []
    for m in methods:
        e = [err(r.get(m, np.nan), r["true_period"]) for r in records
             if np.isfinite(r.get(m, np.nan)) and r.get(m, 0) > 0]
        if e:
            meds.append(np.median(e)); labs.append(m); cols.append(COLORS.get(m, "gray"))
    ax.bar(range(len(meds)), meds, color=cols)
    for i, v in enumerate(meds):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=20, ha="right")
    ax.set_ylabel("median |error| [%]"); ax.set_title("Lower is better")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(B.FIGURES / "fig02_median_error.png"); written_figs.append("fig02_median_error.png")
    plt.close(fig)

    # fig04: continuous gain-vs-b scatter with Spearman (the primary result)
    try:
        from scipy.stats import spearmanr
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for m in [x for x in model_methods if x != "analytic"]:
            bs, gains = [], []
            for r in records:
                b = r.get("known_b")
                va, vm = r.get("analytic", np.nan), r.get(m, np.nan)
                if b is None or not np.isfinite(b):
                    continue
                if not (np.isfinite(va) and va > 0 and np.isfinite(vm) and vm > 0):
                    continue
                bs.append(b)
                gains.append(err(va, r["true_period"]) - err(vm, r["true_period"]))
            if len(bs) < 5:
                continue
            rho, pv = spearmanr(bs, gains)
            ax.scatter(bs, gains, s=42, alpha=0.85, label=f"{m}  rho={rho:+.2f}, p={pv:.3f}",
                       color=COLORS.get(m, "gray"), edgecolor="k", linewidth=0.3)
        ax.axhline(0, color="k", lw=1)
        ax.axvline(1.0, ls=":", color="gray", lw=1)
        ax.text(1.005, ax.get_ylim()[1]*0.95, "b>1", fontsize=7, color="gray")
        ax.set_xlabel("published impact parameter b")
        ax.set_ylabel("analytic error - model error  [percentage points]")
        ax.set_title("Does the model beat the b=0 analytic estimator more as b rises?")
        ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(B.FIGURES / "fig04_gain_vs_b.png"); written_figs.append("fig04_gain_vs_b.png")
        plt.close(fig)
    except Exception:
        pass

    # fig03 breakdown (eccentricity) + reference binned b
    for key, bins, name, fname in [("known_ecc", EBINS, "eccentricity", "fig03_by_eccentricity"),
                                   ("known_b", BBINS, "impact parameter", "fig04b_binned_b_reference")]:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        width = 0.8 / max(len(model_methods), 1)
        xs = np.arange(len(bins))
        for j, m in enumerate(model_methods):
            vals, ns = [], []
            for lo, hi, _ in bins:
                sel = [r for r in records if r.get(key) is not None
                       and np.isfinite(r.get(key, np.nan)) and lo <= r[key] < hi]
                e = [err(r.get(m, np.nan), r["true_period"]) for r in sel
                     if np.isfinite(r.get(m, np.nan)) and r.get(m, 0) > 0]
                vals.append(np.median(e) if e else np.nan); ns.append(len(sel))
            ax.bar(xs + j * width, vals, width, label=m, color=COLORS.get(m, "gray"))
        ax.set_xticks(xs + 0.4 - width / 2)
        ax.set_xticklabels([f"{nm}\n(n={sum(1 for r in records if r.get(key) is not None and np.isfinite(r.get(key, np.nan)) and lo <= r[key] < hi)})"
                            for lo, hi, nm in bins], fontsize=8)
        ax.set_ylabel("median |error| [%]"); ax.set_xlabel(name)
        ax.set_title(f"Real data: error vs {name}")
        ax.legend(fontsize=7.5); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(B.FIGURES / f"{fname}.png"); written_figs.append(f"{fname}.png")
        plt.close(fig)

    # fig05 coverage / intervals
    interval_methods = [m for m in model_methods if any(m + "_lo" in r for r in records)]
    if interval_methods:
        fig, ax = plt.subplots(figsize=(8, 5.6))
        order = sorted(records, key=lambda x: x["true_period"])
        y = np.arange(len(order))
        m = interval_methods[-1]
        for r, yy in zip(order, y):
            if m + "_lo" not in r:
                continue
            inside = r[m + "_lo"] <= r["true_period"] <= r[m + "_hi"]
            ax.plot([r[m + "_lo"], r[m + "_hi"]], [yy, yy], "-",
                    color="#2e8b57" if inside else "#b5442f", lw=3, alpha=0.55)
            ax.plot(r[m], yy, "o", color="#2e8b57" if inside else "#b5442f", ms=5)
            ax.plot(r["true_period"], yy, "k*", ms=10)
        ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in order], fontsize=6.5)
        ax.set_xscale("log"); ax.set_xlabel("period [d]")
        ax.set_title(f"1-sigma intervals, {m} (star = truth; green = covered)")
        ax.grid(alpha=0.25, axis="x", which="both")
        fig.tight_layout(); fig.savefig(B.FIGURES / "fig05_intervals.png"); written_figs.append("fig05_intervals.png")
        plt.close(fig)

    print("\n" + "=" * 108)
    print("FIGURE FILES WRITTEN (use these exact names in LaTeX \\includegraphics)")
    print("=" * 108)
    for f in written_figs:
        full = B.FIGURES / f
        print(f"  {f:38s} {full}  ({full.stat().st_size // 1024} KB)"
              if full.exists() else f"  {f:38s} MISSING")
    print(f"\n[14] figure directory: {B.FIGURES}")
    with open(B.RESULTS / "FIGURE_LIST.txt", "w") as fh:
        fh.write("\n".join(written_figs) + "\n")
    print(f"[14] figure list -> {B.RESULTS / 'FIGURE_LIST.txt'}")


if __name__ == "__main__":
    main()
