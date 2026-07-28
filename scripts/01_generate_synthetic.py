#!/usr/bin/env python
"""Generate the synthetic single-transit training set.

    python scripts/01_generate_synthetic.py --n 40000
"""

import argparse

import _bootstrap as B
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exopinn import synth
from exopinn.lightcurve import WINDOW_DAYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(B.DATA / "synthetic_transits.npz"))
    args = ap.parse_args()

    cfg = synth.SynthConfig(n_samples=args.n, seed=args.seed)
    data = synth.generate(cfg)
    synth.save(data, args.out)
    print(f"[01] wrote {args.out}  lc{data['lc'].shape}  scalars{data['scalars'].shape}")

    # Sanity figure: a handful of examples across the period range, plus coverage.
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    t = np.linspace(-WINDOW_DAYS / 2, WINDOW_DAYS / 2, data["lc"].shape[1])
    order = np.argsort(data["period"])
    picks = order[np.linspace(0, len(order) - 1, 8).astype(int)]
    for ax, i in zip(axes.ravel(), picks):
        ax.plot(t, data["lc"][i], ".-", ms=2, lw=0.7)
        ax.set_title(f"P={data['period'][i]:.1f}d  D={data['scalars'][i,0]:.1f}h\n"
                     f"depth={data['scalars'][i,1]*100:.2f}%  e={data['meta'][i,2]:.2f}",
                     fontsize=8)
        ax.set_xlabel("days from mid-transit", fontsize=7)
    fig.suptitle("Synthetic single transits (fixed 2.5-day absolute-time window)")
    fig.tight_layout()
    fig.savefig(B.FIGURES / "synthetic_examples.png", dpi=140)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(np.log10(data["period"]), bins=60, color="#3b6ea5")
    axes[0].set_xlabel("log10 period [d]"); axes[0].set_ylabel("count")
    axes[1].scatter(data["scalars"][:, 0], data["period"], s=2, alpha=0.15, c=data["meta"][:, 2], cmap="magma")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("measured duration [h]"); axes[1].set_ylabel("true period [d]")
    axes[1].set_title("colour = eccentricity")
    axes[2].scatter(data["scalars"][:, 1] * 100, data["meta"][:, 1], s=2, alpha=0.15)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("measured depth [%]"); axes[2].set_ylabel("impact parameter b")
    fig.tight_layout()
    fig.savefig(B.FIGURES / "synthetic_coverage.png", dpi=140)
    print(f"[01] figures -> {B.FIGURES}")


if __name__ == "__main__":
    main()
