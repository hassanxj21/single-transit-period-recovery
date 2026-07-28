#!/usr/bin/env python
"""Download a pool of real TESS light curves to inject synthetic transits into.

These supply what the synthetic generator cannot: real photon and systematic
noise, real cadence, real momentum-dump and scattered-light residuals, and real
gaps. Injecting into them BEFORE detrending means the injected transit is
distorted by the same Savitzky-Golay filter that distorts a real one - which is
the domain gap that broke the CNN branch.

Two properties this pool must have, and an earlier version of this script had
neither:

* **No overlap with the evaluation targets.** Every star in ``targets.py`` is
  excluded. Training on noise drawn from the same stars the model is scored on
  is leakage: it lets the network adapt to those specific systematics.
* **Diversity in brightness and stellar type.** Noise amplitude scales with
  magnitude, so a pool drawn from a handful of stars teaches the model one
  noise regime. Stars are drawn from several sky regions and explicit Tmag
  bins.

    python scripts/09_build_noise_pool.py --n-lightcurves 80
"""

import argparse
import warnings

import _bootstrap as B
import numpy as np

from exopinn.data import download_lightcurve, find_transit, search_products
from exopinn.targets import ALL

warnings.filterwarnings("ignore")

# Never inject into a star we evaluate on.
EXCLUDE = set(ALL.keys())

# Regions spread over sky position so sector coverage and systematics vary.
REGIONS = ["90.0 -66.0", "270.0 66.0", "180.0 -45.0"]
TMAG_BINS = [(8.0, 10.0), (10.0, 12.5)]

# Each TIC cone search takes ~40 s, so the star list is cached. Losing the
# download loop should never mean re-paying for the catalogue queries.
TIC_CACHE = B.DATA / "pool_tic_list.json"


def candidate_tics(per_bin=14):
    """Field stars across several regions and brightness bins."""
    import json

    if TIC_CACHE.exists():
        cached = json.loads(TIC_CACHE.read_text())
        print(f"[09] reusing cached star list ({len(cached)} stars) from {TIC_CACHE}", flush=True)
        return [(int(a), float(b)) for a, b in cached]

    from astroquery.mast import Catalogs

    out = []
    n_q = len(REGIONS) * len(TMAG_BINS)
    for i, coord in enumerate(REGIONS):
        for j, (lo, hi) in enumerate(TMAG_BINS):
            q = i * len(TMAG_BINS) + j + 1
            print(f"[09] TIC query {q}/{n_q}: {coord} Tmag {lo}-{hi} (~40 s)...", flush=True)
            try:
                cat = Catalogs.query_criteria(catalog="TIC", coordinates=coord, radius=0.8,
                                              Tmag=[lo, hi], objType="STAR")
            except Exception as exc:
                print(f"       failed: {type(exc).__name__}", flush=True)
                continue
            if len(cat) == 0:
                print("       0 rows", flush=True)
                continue
            df = cat.to_pandas().dropna(subset=["Tmag"]).sort_values("Tmag")
            added = 0
            for tid, tmag in zip(df["ID"].astype(int), df["Tmag"]):
                if added >= per_bin:
                    break
                if int(tid) not in EXCLUDE:
                    out.append((int(tid), float(tmag)))
                    added += 1
            print(f"       {len(cat)} rows -> kept {added}", flush=True)

    seen, uniq = set(), []
    for tid, tmag in out:
        if tid not in seen:
            seen.add(tid)
            uniq.append((tid, tmag))
    TIC_CACHE.write_text(json.dumps(uniq))
    print(f"[09] cached {len(uniq)} stars -> {TIC_CACHE}", flush=True)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lightcurves", type=int, default=80)
    ap.add_argument("--max-per-star", type=int, default=2)
    ap.add_argument("--out", default=str(B.DATA / "noise_pool.npz"))
    args = ap.parse_args()

    tics = candidate_tics()
    print(f"[09] {len(tics)} candidate field stars (evaluation targets excluded)\n")

    segments, meta = [], []
    for tic, tmag in tics:
        if len(segments) >= args.n_lightcurves:
            break
        try:
            search = search_products(tic)
            if len(search) == 0:
                continue
            sectors = sorted({int(s) for s in search.table["sequence_number"]})
        except Exception:
            continue

        n_ok = 0
        for sector in sectors:
            if n_ok >= args.max_per_star or len(segments) >= args.n_lightcurves:
                break
            try:
                lc, prov = download_lightcurve(tic, sector=sector)
                clean = lc.remove_nans().normalize()
                t = np.asarray(clean.time.value, float)
                f = np.asarray(clean.flux.value, float)
                ok = np.isfinite(t) & np.isfinite(f)
                t, f = t[ok], f[ok]
                if len(t) < 500:
                    continue

                # Exclude the deepest dip: it could be a real transit or an
                # eclipsing binary, and injected transits must be the only signal.
                hit = find_transit(t, f)
                excl = np.zeros(len(t), bool)
                if hit is not None:
                    excl = np.abs(t - hit[0]) < 1.5

                segments.append(np.column_stack([t, f, excl.astype(float)]))
                meta.append((tic, prov["sector"], prov["exptime"], len(t), tmag))
                n_ok += 1
                print(f"  [{len(segments):>3}/{args.n_lightcurves}] TIC {tic} S{prov['sector']:>3} "
                      f"{prov['author']:<12} {prov['exptime']:>5.0f}s Tmag={tmag:5.2f} "
                      f"n={len(t):>6}{'  [dip masked]' if hit else ''}", flush=True)
            except Exception:
                continue

    if not segments:
        raise SystemExit("no light curves downloaded")

    np.savez_compressed(args.out, **{f"seg{i}": s for i, s in enumerate(segments)},
                        meta=np.array(meta, float))
    m = np.array(meta, float)
    total_days = sum(s[:, 0].max() - s[:, 0].min() for s in segments)
    print(f"\n[09] {len(segments)} light curves from {len(set(m[:,0].astype(int)))} distinct stars")
    print(f"[09] {total_days:.0f} days total, Tmag {m[:,4].min():.1f}-{m[:,4].max():.1f}, "
          f"{len(set(m[:,1].astype(int)))} sectors, cadences {sorted(set(m[:,2].tolist()))}")
    print(f"[09] evaluation-target overlap: {sorted(EXCLUDE & set(m[:,0].astype(int).tolist()))}")
    print(f"[09] wrote {args.out}")


if __name__ == "__main__":
    main()
