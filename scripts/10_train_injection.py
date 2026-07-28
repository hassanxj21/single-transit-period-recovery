#!/usr/bin/env python
"""Build the injection-recovery training set, train on it, and re-test on real data.

This is the test of whether the CNN branch's advantage was an artefact of the
simulator. On purely synthetic light curves the CNN branch beat a scalars-only
model by 28%; on real TESS targets it lost to that same model by 3.6x. If the
gap was a domain gap, training on synthetic transits injected into *real* TESS
photometry should close it. If the CNN still loses here, the honest conclusion
is that transit shape as measured by TESS does not carry usable impact-parameter
information at these signal-to-noise ratios.

    python scripts/10_train_injection.py --n 30000 --epochs 60
"""

import argparse

import _bootstrap as B
import numpy as np

from exopinn import injection, synth
from exopinn.train import TrainConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(B.DATA / "noise_pool.npz"))
    ap.add_argument("--out", default=str(B.DATA / "injected.npz"))
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    if not args.skip_build:
        pool = injection.load_pool(args.pool)
        spans = [s[:, 0].max() - s[:, 0].min() for s in pool]
        print(f"[10] noise pool: {len(pool)} light curves, {sum(spans):.0f} days of real photometry")
        cfg = injection.InjectionConfig(n_samples=args.n, seed=args.seed)
        data = injection.generate(pool, cfg)
        synth.save(data, args.out)
        print(f"[10] wrote {args.out}\n")
    else:
        data = synth.load(args.out)
        print(f"[10] loaded {len(data['period']):,} injected samples\n")

    for tag, use_lc in [("inj_full", True), ("inj_scalars", False)]:
        print(f"=== training {tag} (use_lightcurve={use_lc}) ===")
        train(data, TrainConfig(epochs=args.epochs, use_lightcurve=use_lc),
              out_dir=B.MODELS / f"cnn_pinn_{tag}")
        print()

    print("[10] done. Evaluate on real targets with scripts/11_final_comparison.py")


if __name__ == "__main__":
    main()
