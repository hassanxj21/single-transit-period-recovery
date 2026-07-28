#!/usr/bin/env python
"""Does the CNN-PINN need more training, or has it hit the information limit?

"The loss is still going down" is the wrong test. This problem has an
**irreducible noise floor**: at fixed observables, the true period is genuinely
ambiguous, because eccentricity and impact parameter are unobservable from one
transit and both rescale the duration. No amount of training removes that.

So the diagnostic compares three numbers:

  A. the model's actual residual scatter on held-out data
  B. the SCALAR floor  - posterior width given only (duration, depth, M*, R*),
     marginalising over the b and e priors. This is the best any method using
     those four numbers can do.
  C. the SHAPE floor   - the same, but with b known exactly. The light curve
     constrains b through the ingress slope and the V-vs-U profile, so C is the
     best the CNN branch could achieve with a perfect read of the shape.

Verdicts:
  scatter >> B   -> underfit; more epochs or capacity will help
  scatter ~= B   -> converged against the scalars; more epochs will NOT help.
                    The only remaining headroom is the CNN branch, bounded by C.
  scatter <  B   -> the model is extracting shape information (the CNN is
                    earning its place); compare against C for remaining room.

    python scripts/07_convergence_diagnostic.py
"""

import argparse
import json

import _bootstrap as B
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

from exopinn import synth
from exopinn.evaluate import analytic_periods, coverage, metrics
from exopinn.physics import period_from_duration
from exopinn.train import load_model, predict

ECC_PRIOR = (0.867, 3.03)


def posterior_width(duration_hr, depth, mass, radius, n_mc, rng, fixed_b=None):
    """Spread (in dex) of periods consistent with these observables."""
    k = np.sqrt(max(depth, 1e-8))
    b = rng.uniform(0.0, 1.0 + k, n_mc) if fixed_b is None else np.full(n_mc, fixed_b)
    e = np.minimum(beta_dist.rvs(*ECC_PRIOR, size=n_mc, random_state=rng), 0.9)
    om = rng.uniform(0.0, 2 * np.pi, n_mc)

    out = np.empty(n_mc)
    for i in range(n_mc):
        out[i] = period_from_duration(duration_hr, mass, radius, k=k, b=b[i],
                                      ecc=e[i], omega=om[i], contact="fwhm")
    good = out[np.isfinite(out) & (out > 0)]
    if len(good) < 20:
        return np.nan
    lo, hi = np.percentile(np.log10(good), [16, 84])
    return (hi - lo) / 2.0  # 1-sigma half-width in dex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(B.MODELS / "cnn_pinn" / "cnn_pinn.pt"))
    ap.add_argument("--data", default=str(B.DATA / "synthetic_transits.npz"))
    ap.add_argument("--n-floor", type=int, default=250, help="val samples used for the floor estimate")
    ap.add_argument("--n-mc", type=int, default=400)
    args = ap.parse_args()

    model, norm, ckpt = load_model(args.model)
    hist = json.loads((B.MODELS / "cnn_pinn" / "history.json").read_text())
    data = synth.load(args.data)
    va = ckpt["val_indices"]
    lc, sc, p_true = data["lc"][va], data["scalars"][va], data["period"][va]
    b_true = data["meta"][va, 1]

    # ---------------------------------------------------------- 1. converged?
    print("=" * 84)
    print("1. Is the optimisation still improving?")
    print("=" * 84)
    v = np.asarray(hist["val_nll"], float)
    n = len(v)
    tail = v[int(0.8 * n):]
    slope = np.polyfit(np.arange(len(tail)), tail, 1)[0]
    print(f"  epochs run                    {n}")
    print(f"  val NLL  first / min / last   {v[0]:.4f} / {v.min():.4f} / {v[-1]:.4f}")
    print(f"  best epoch                    {int(np.argmin(v))} of {n}")
    print(f"  slope over final 20% of run   {slope:+.2e} per epoch")
    improved_late = (v.min() < np.min(v[:int(0.8 * n)]) - 1e-4)
    print(f"  best epoch in final 20%?      {'yes' if int(np.argmin(v)) > 0.8 * n else 'no'}")
    print(f"  -> optimisation {'has NOT plateaued' if slope < -1e-4 else 'has plateaued'}")

    tr = np.asarray(hist["train_data"], float)
    print(f"  train/val gap at end          {v[-1] - tr[-1]:+.4f} "
          f"({'overfitting' if v[-1] - tr[-1] > 0.1 else 'no meaningful overfit'})")

    # ------------------------------------------- 2. model vs analytic vs floor
    print()
    print("=" * 84)
    print("2. Model scatter vs the information floor")
    print("=" * 84)
    p_pinn, lo, hi = predict(model, norm, lc, sc)
    p_ana = analytic_periods(sc)
    m_pinn, m_ana = metrics(p_pinn, p_true), metrics(p_ana, p_true)
    cov = coverage(lo, hi, p_true)

    resid = np.log10(p_pinn / p_true)
    scatter = float(np.percentile(np.abs(resid), 68))
    resid_a = np.log10(p_ana / p_true)
    scatter_a = float(np.percentile(np.abs(resid_a[np.isfinite(resid_a)]), 68))

    rng = np.random.default_rng(11)
    idx = rng.choice(len(p_true), size=min(args.n_floor, len(p_true)), replace=False)
    floor_scalar, floor_shape = [], []
    for i in idx:
        d, dep, m_, r_ = sc[i]
        floor_scalar.append(posterior_width(d, dep, m_, r_, args.n_mc, rng))
        floor_shape.append(posterior_width(d, dep, m_, r_, args.n_mc, rng, fixed_b=b_true[i]))
    fs = float(np.nanmedian(floor_scalar))
    fh = float(np.nanmedian(floor_shape))

    print(f"  A. CNN-PINN residual scatter (68th pct |log10 err|)   {scatter:.3f} dex "
          f"= x{10**scatter:.2f}")
    print(f"     analytic inversion, same measure                   {scatter_a:.3f} dex "
          f"= x{10**scatter_a:.2f}")
    print(f"  B. SCALAR floor (marginalise b and e)                 {fs:.3f} dex "
          f"= x{10**fs:.2f}")
    print(f"  C. SHAPE floor  (b known exactly, marginalise e)      {fh:.3f} dex "
          f"= x{10**fh:.2f}")
    print()
    print(f"  model / scalar floor = {scatter / fs:.2f}")
    print(f"  headroom the CNN branch could still win (B - C) = {fs - fh:.3f} dex "
          f"= a factor of {10**(fs - fh):.2f}")

    print()
    if scatter > 1.6 * fs:
        verdict = ("UNDERFIT. The model is well above the floor set by its own inputs, so\n"
                   "  more epochs, more capacity or more data should still help.")
    elif scatter > 1.05 * fs:
        verdict = ("CONVERGED against the scalar features. More epochs will not help - the\n"
                   "  residual is dominated by the b/e ambiguity, not by optimisation error.\n"
                   "  Remaining headroom is entirely in the CNN branch reading b off the shape.")
    else:
        verdict = ("BEATING the scalar floor - the CNN branch is extracting shape information\n"
                   "  the analytic inversion cannot access. This is the result that justifies\n"
                   "  the architecture. Compare against C for what is still left.")
    print(f"  VERDICT: {verdict}")

    # ------------------------------------------------------- 3. calibration
    print()
    print("=" * 84)
    print("3. Are the predicted intervals honest?")
    print("=" * 84)
    print(f"  1-sigma coverage            {cov['coverage']:.1%}   (target 68%)")
    print(f"  median interval half-width  {cov['median_width_dex']/2:.3f} dex "
          f"= x{10**(cov['median_width_dex']/2):.2f}")
    print(f"  actual residual scatter     {scatter:.3f} dex")
    ratio = (cov["median_width_dex"] / 2) / scatter
    print(f"  predicted / actual          {ratio:.2f}  "
          f"({'over-confident' if ratio < 0.8 else 'under-confident' if ratio > 1.25 else 'well calibrated'})")

    print()
    print(f"  median |error|   analytic {m_ana['median_pct']:.1f}%   CNN-PINN {m_pinn['median_pct']:.1f}%")
    print(f"  within factor 2  analytic {m_ana['frac_within_factor_2']:.0%}   "
          f"CNN-PINN {m_pinn['frac_within_factor_2']:.0%}")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(hist["epoch"], hist["val_nll"], label="val NLL")
    axes[0].plot(hist["epoch"], hist["train_data"], label="train NLL", alpha=0.7)
    axes[0].axvline(int(np.argmin(v)), ls=":", color="k", label=f"best epoch {int(np.argmin(v))}")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Gaussian NLL"); axes[0].legend(fontsize=8)
    axes[0].set_title("Optimisation"); axes[0].grid(alpha=0.3)

    axes[1].bar(["CNN-PINN", "analytic", "scalar\nfloor", "shape\nfloor"],
                [scatter, scatter_a, fs, fh],
                color=["#2e8b57", "#3b6ea5", "#888", "#ccc"])
    axes[1].set_ylabel("1-sigma scatter [dex]")
    axes[1].set_title("Error vs information floor")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].hist(np.clip(resid, -1.5, 1.5), bins=70, density=True, alpha=0.8,
                 color="#2e8b57", label="CNN-PINN")
    axes[2].hist(np.clip(resid_a[np.isfinite(resid_a)], -1.5, 1.5), bins=70, density=True,
                 alpha=0.45, color="#3b6ea5", label="analytic")
    axes[2].axvline(0, color="k", lw=1)
    axes[2].set_xlabel("log10(predicted / true)"); axes[2].legend(fontsize=8)
    axes[2].set_title("Residuals")
    fig.tight_layout(); fig.savefig(B.FIGURES / "convergence_diagnostic.png", dpi=150)
    print(f"\n  figure -> {B.FIGURES / 'convergence_diagnostic.png'}")


if __name__ == "__main__":
    main()
