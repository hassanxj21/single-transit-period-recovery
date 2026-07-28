#!/usr/bin/env python
"""Train the hybrid CNN-PINN and benchmark it against the analytic inversion.

    python scripts/02_train_pinn.py --epochs 400
    python scripts/02_train_pinn.py --lambda-physics 0.0   # ablation
"""

import argparse
import json

import _bootstrap as B
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exopinn import synth
from exopinn.evaluate import analytic_periods, coverage, format_table, metrics
from exopinn.train import TrainConfig, predict, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(B.DATA / "synthetic_transits.npz"))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--lambda-physics", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    out_dir = B.MODELS / f"cnn_pinn{tag}"

    data = synth.load(args.data)
    print(f"[02] {len(data['period']):,} samples, light curves {data['lc'].shape}")

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, lambda_physics=args.lambda_physics)
    model, norm, history, (tr_idx, va_idx) = train(data, cfg, out_dir=out_dir)

    # ---------------- evaluation on the held-out split ----------------
    lc_va, sc_va, p_va = data["lc"][va_idx], data["scalars"][va_idx], data["period"][va_idx]
    ecc_va = data["meta"][va_idx, 2]

    p_pinn, lo, hi = predict(model, norm, lc_va, sc_va)
    p_ana = analytic_periods(sc_va)

    m_pinn, m_ana = metrics(p_pinn, p_va), metrics(p_ana, p_va)
    cov = coverage(lo, hi, p_va)

    keys = ["n", "n_failed", "median_pct", "mean_pct", "median_dex",
            "frac_within_25pct", "frac_within_factor_2", "median_bias_ratio"]
    rows = [[k, f"{m_ana[k]:.4g}", f"{m_pinn[k]:.4g}"] for k in keys]
    print("\n[02] held-out synthetic performance")
    print(format_table(rows, ["metric", "analytic inversion", "CNN-PINN"]))
    print(f"\n[02] 1-sigma interval coverage {cov['coverage']:.1%} "
          f"(target 68%), median width {cov['median_width_dex']:.3f} dex "
          f"= x{10**cov['median_width_dex']:.2f}")

    # Split by eccentricity: this is where the physics assumption breaks.
    print("\n[02] median |error| by eccentricity")
    ebins = [(0.0, 1e-6), (1e-6, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.9)]
    rows = []
    for a, b in ebins:
        sel = (ecc_va >= a) & (ecc_va < b)
        if sel.sum() < 20:
            continue
        rows.append([f"{a:.2f}-{b:.2f}" if b < 1 else f">{a}", int(sel.sum()),
                     f"{metrics(p_ana[sel], p_va[sel])['median_pct']:.1f}%",
                     f"{metrics(p_pinn[sel], p_va[sel])['median_pct']:.1f}%"])
    print(format_table(rows, ["eccentricity", "n", "analytic", "CNN-PINN"]))

    summary = {"config": cfg.to_dict(), "analytic": m_ana, "cnn_pinn": m_pinn, "coverage": cov}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(history["epoch"], history["train_data"], label="data (Gaussian NLL)")
    axes[0].plot(history["epoch"], history["train_physics"], label="physics")
    axes[0].plot(history["epoch"], history["val_nll"], label="val NLL", ls="--")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("CNN-PINN training")
    axes[1].plot(history["epoch"], history["val_median_pct"], color="#b5442f")
    axes[1].axhline(m_ana["median_pct"], ls=":", color="k",
                    label=f"analytic baseline ({m_ana['median_pct']:.1f}%)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val median |error| [%]")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(B.FIGURES / f"loss_curve{tag}.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    lim = [p_va.min() * 0.7, p_va.max() * 1.4]
    for ax, pred, name, m in [(axes[0], p_ana, "Analytic inversion (circular, b=0)", m_ana),
                              (axes[1], p_pinn, "CNN-PINN", m_pinn)]:
        sc = ax.scatter(p_va, pred, s=4, alpha=0.25, c=ecc_va, cmap="magma", vmin=0, vmax=0.6)
        ax.plot(lim, lim, "k-", lw=1)
        for f, ls in [(2.0, "--"), (0.5, "--")]:
            ax.plot(lim, [f * x for x in lim], ls, color="gray", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("true period [d]")
        ax.set_title(f"{name}\nmedian |err| {m['median_pct']:.1f}%, "
                     f"within 2x: {m['frac_within_factor_2']:.0%}", fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("predicted period [d]")
    fig.colorbar(sc, ax=axes, label="eccentricity", fraction=0.03)
    fig.savefig(B.FIGURES / f"predicted_vs_true{tag}.png", dpi=150, bbox_inches="tight")

    # Calibration: does the predicted sigma mean anything?
    fig, ax = plt.subplots(figsize=(5.5, 5))
    z = np.log10(p_pinn / p_va) / (np.log10(hi / p_pinn) + 1e-12)
    ax.hist(np.clip(z, -5, 5), bins=80, density=True, color="#3b6ea5", alpha=0.8)
    xs = np.linspace(-5, 5, 400)
    ax.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "k-", lw=1.5, label="unit normal")
    ax.set_xlabel("(log10 pred - log10 true) / predicted sigma")
    ax.set_title(f"Uncertainty calibration\n1-sigma coverage {cov['coverage']:.1%} (target 68%)")
    ax.legend(); fig.tight_layout()
    fig.savefig(B.FIGURES / f"calibration{tag}.png", dpi=150)

    print(f"\n[02] figures -> {B.FIGURES}")
    print(f"[02] model    -> {out_dir / 'cnn_pinn.pt'}")


if __name__ == "__main__":
    main()
