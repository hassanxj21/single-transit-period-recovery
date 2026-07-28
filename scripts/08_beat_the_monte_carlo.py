#!/usr/bin/env python
"""Does the CNN-PINN beat the Monte Carlo? Trains both models and settles it.

Four methods, all evaluated on the same held-out split:

  1. analytic inversion      closed form, circular, b=0. No uncertainty.
  2. kNN Bayes posterior     the Monte Carlo done exactly right. The training set
                             IS a Monte Carlo sample from the joint prior, so the
                             posterior given the four scalars is a rejection-
                             sampling lookup in it. Same inputs, same priors,
                             including the period prior. This is the number to beat.
  3. CNN-PINN, scalars only  the ablation. Same network, CNN branch removed. If
                             this matches (2), the network has simply learned the
                             posterior - correct, but not evidence for the CNN.
  4. CNN-PINN, full          if this beats (2) AND (3), the light curve is
                             supplying information the scalars cannot, which is
                             the claim the architecture rests on.

    python scripts/08_beat_the_monte_carlo.py
    python scripts/08_beat_the_monte_carlo.py --epochs 60 --skip-training
"""

import argparse
import json

import _bootstrap as B
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exopinn import synth
from exopinn.evaluate import KNNPosterior, analytic_periods, coverage, format_table, metrics
from exopinn.train import TrainConfig, load_model, predict, train


def scatter_dex(pred, true):
    r = np.log10(np.asarray(pred, float) / np.asarray(true, float))
    r = r[np.isfinite(r)]
    return float(np.percentile(np.abs(r), 68))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(B.DATA / "synthetic_v2.npz"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--knn-k", type=int, default=250)
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()

    data = synth.load(args.data)
    print(f"[08] {len(data['period']):,} samples from {args.data}\n")

    runs = {}
    for tag, use_lc in [("full", True), ("scalars_only", False)]:
        out_dir = B.MODELS / f"cnn_pinn_{tag}"
        if not args.skip_training:
            print(f"=== training: {tag} (use_lightcurve={use_lc}) ===")
            cfg = TrainConfig(epochs=args.epochs, use_lightcurve=use_lc)
            train(data, cfg, out_dir=out_dir)
            print()
        runs[tag] = load_model(out_dir / "cnn_pinn.pt")

    # Evaluate everything on the split the FULL model held out, so the kNN
    # lookup never sees a validation sample as one of its own neighbours.
    _, _, ckpt = runs["full"]
    va = ckpt["val_indices"]
    tr_mask = np.ones(len(data["period"]), bool)
    tr_mask[va] = False
    tr_idx = np.flatnonzero(tr_mask)

    sc_va, p_va = data["scalars"][va], data["period"][va]
    lc_va = data["lc"][va]
    ecc_va = data["meta"][va, 2]

    results = {}

    results["analytic"] = (analytic_periods(sc_va), None, None)

    knn = KNNPosterior(data["scalars"][tr_idx], data["period"][tr_idx], k=args.knn_k)
    results["knn_bayes"] = knn.predict(sc_va)

    for tag in ("scalars_only", "full"):
        model, norm, _ = runs[tag]
        results[f"pinn_{tag}"] = predict(model, norm, lc_va, sc_va)

    names = {
        "analytic": "1. analytic inversion",
        "knn_bayes": "2. kNN Bayes (the Monte Carlo)",
        "pinn_scalars_only": "3. CNN-PINN, scalars only",
        "pinn_full": "4. CNN-PINN, full",
    }

    print("=" * 92)
    print("HEAD TO HEAD on held-out synthetic data")
    print("=" * 92)
    rows = []
    for key, label in names.items():
        pred, lo, hi = results[key]
        m = metrics(pred, p_va)
        s = scatter_dex(pred, p_va)
        cov = coverage(lo, hi, p_va) if lo is not None else None
        rows.append([
            label, f"{m['median_pct']:.1f}%", f"{s:.3f}", f"x{10**s:.2f}",
            f"{m['frac_within_factor_2']:.0%}", f"{m['median_bias_ratio']:.2f}",
            "-" if cov is None else f"{cov['coverage']:.0%}",
        ])
    print(format_table(rows, ["method", "median err", "scatter [dex]", "as factor",
                              "within 2x", "bias", "1sig cov"]))

    s_knn = scatter_dex(results["knn_bayes"][0], p_va)
    s_sc = scatter_dex(results["pinn_scalars_only"][0], p_va)
    s_full = scatter_dex(results["pinn_full"][0], p_va)

    print()
    print("-" * 92)
    print("VERDICT")
    print("-" * 92)
    print(f"  kNN Bayes (Monte Carlo)  {s_knn:.3f} dex")
    print(f"  CNN-PINN scalars only    {s_sc:.3f} dex   ({s_sc / s_knn:.2f}x the Monte Carlo)")
    print(f"  CNN-PINN full            {s_full:.3f} dex   ({s_full / s_knn:.2f}x the Monte Carlo)")
    print()
    if s_sc > 1.15 * s_knn:
        print("  The scalars-only network is WORSE than the kNN posterior, so it has not")
        print("  fully learned the scalar mapping - the comparison below is confounded.")
    else:
        print("  The scalars-only network matches the kNN posterior, confirming the network")
        print("  does learn the correct scalar posterior. So it is a clean control.")
    gain = (s_sc - s_full) / s_sc if s_sc > 0 else 0.0
    print()
    if s_full < 0.95 * s_sc:
        print(f"  The full model is {gain:.0%} tighter than the scalars-only ablation.")
        print("  That gap cannot come from the period prior (both models have it) and cannot")
        print("  come from the four scalars (both models see them). It can ONLY come from the")
        print("  light curve - the CNN is reading transit shape, most plausibly impact")
        print("  parameter via the ingress slope and the V-vs-U profile.")
    else:
        print("  The full model does NOT improve on the scalars-only ablation. The CNN branch")
        print("  is not currently extracting usable shape information, and the apparent")
        print("  advantage over a marginalised baseline is period-prior shrinkage instead.")

    if s_full < s_knn:
        print()
        print(f"  >>> The CNN-PINN BEATS the Monte Carlo by a factor of {s_knn / s_full:.2f}"
              f" in scatter. <<<")

    print("\nby eccentricity (median |error|)")
    ebins = [(0.0, 1e-9), (1e-9, 0.15), (0.15, 0.35), (0.35, 0.9)]
    rows = []
    for a, b in ebins:
        sel = (ecc_va >= a) & (ecc_va < b)
        if sel.sum() < 30:
            continue
        rows.append([f"{a:.2f}-{b:.2f}", int(sel.sum())] +
                    [f"{metrics(results[k][0][sel], p_va[sel])['median_pct']:.1f}%" for k in names])
    print(format_table(rows, ["ecc", "n"] + [n.split(". ")[1] for n in names.values()]))

    summary = {k: {"median_pct": metrics(results[k][0], p_va)["median_pct"],
                   "scatter_dex": scatter_dex(results[k][0], p_va)} for k in names}
    (B.DATA / "beat_the_mc.json").write_text(json.dumps(summary, indent=2))

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    labels = [names[k].split(". ")[1] for k in names]
    vals = [scatter_dex(results[k][0], p_va) for k in names]
    colors = ["#3b6ea5", "#888888", "#c08a3e", "#2e8b57"]
    axes[0].bar(range(4), vals, color=colors)
    axes[0].axhline(s_knn, ls="--", color="k", lw=1, label="Monte Carlo")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels([l.replace(", ", ",\n").replace(" (", "\n(") for l in labels], fontsize=8)
    axes[0].set_ylabel("1-sigma scatter [dex]")
    axes[0].set_title("Lower is better"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    lim = [p_va.min() * 0.7, p_va.max() * 1.4]
    axes[1].scatter(p_va, results["knn_bayes"][0], s=3, alpha=0.2, c="#888888", label="Monte Carlo")
    axes[1].scatter(p_va, results["pinn_full"][0], s=3, alpha=0.2, c="#2e8b57", label="CNN-PINN full")
    axes[1].plot(lim, lim, "k-", lw=1)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlim(lim); axes[1].set_ylim(lim)
    axes[1].set_xlabel("true period [d]"); axes[1].set_ylabel("predicted period [d]")
    axes[1].legend(fontsize=8, markerscale=3); axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(B.FIGURES / "beat_the_monte_carlo.png", dpi=150)
    print(f"\n[08] figure -> {B.FIGURES / 'beat_the_monte_carlo.png'}")


if __name__ == "__main__":
    main()
