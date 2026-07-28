#!/usr/bin/env python
"""Select an expanded confirmed single-transit validation set from T0's output.

Selection priorities, in order:
  1. published impact parameter AND eccentricity  (lets us test the b/e story)
  2. spread in period from ~26 d upward
  3. spread in eccentricity, deliberately including the high-e tail
  4. spread in transit depth and TESS magnitude
  5. SPOC 120 s data where available (better cadence -> better shape)

Every accept and reject is reported with its reason.

    python scripts/12_select_targets.py --n 28
"""

import argparse
import csv
import warnings

import _bootstrap as B
import numpy as np

from exopinn.data import fetch_tic_properties, search_products
from exopinn.targets import ALL

warnings.filterwarnings("ignore")

QUALIFYING = B.DATA / "t0_qualifying.csv"
OUT = B.DATA / "expanded_targets.csv"


def load_rows():
    rows = []
    for r in csv.DictReader(QUALIFYING.open()):
        def f(k):
            v = r.get(k, "")
            try:
                return float(v) if v not in ("", "None", "nan") else None
            except ValueError:
                return None
        rows.append({"pl_name": r["pl_name"], "tic": int(r["tic_id"]),
                     "pl_orbper": f("pl_orbper"), "pl_trandur": f("pl_trandur"),
                     "pl_imppar": f("pl_imppar"), "pl_orbeccen": f("pl_orbeccen"),
                     "st_mass": f("st_mass"), "st_rad": f("st_rad"), "st_teff": f("st_teff")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=28)
    args = ap.parse_args()

    rows = load_rows()
    print(f"[12] {len(rows)} qualifying planets from T0\n")

    rejected = []
    pool = []
    for r in rows:
        tic = r["tic"]
        if tic in ALL:
            rejected.append((r["pl_name"], "already in target list"))
            continue
        if r["pl_orbper"] is None or r["pl_trandur"] is None:
            rejected.append((r["pl_name"], "missing period or duration"))
            continue
        if r["st_mass"] is None or r["st_rad"] is None:
            rejected.append((r["pl_name"], "missing stellar mass/radius"))
            continue
        if r["pl_imppar"] is None:
            rejected.append((r["pl_name"], "no published impact parameter"))
            continue
        pool.append(r)

    print(f"[12] {len(pool)} have published b; {len(rejected)} rejected on metadata\n")

    # Stratify: guarantee the high-e tail and a period spread rather than
    # letting a naive sort fill the sample with 26-30 d near-circular planets.
    with_e = [r for r in pool if r["pl_orbeccen"] is not None]
    high_e = sorted([r for r in with_e if r["pl_orbeccen"] >= 0.35],
                    key=lambda r: -r["pl_orbeccen"])
    mid_e = sorted([r for r in with_e if 0.15 <= r["pl_orbeccen"] < 0.35],
                   key=lambda r: -r["pl_orbeccen"])
    low_e = sorted([r for r in with_e if r["pl_orbeccen"] < 0.15],
                   key=lambda r: -r["pl_orbper"])
    no_e = sorted([r for r in pool if r["pl_orbeccen"] is None],
                  key=lambda r: -r["pl_orbper"])

    quota = [("high e (>=0.35)", high_e, max(6, args.n // 4)),
             ("mid e (0.15-0.35)", mid_e, max(5, args.n // 5)),
             ("low e (<0.15), longest P", low_e, max(8, args.n // 3)),
             ("no published e, longest P", no_e, args.n)]

    picked, seen = [], set()
    for label, group, cap in quota:
        took = 0
        for r in group:
            if len(picked) >= args.n or took >= cap:
                break
            if r["tic"] in seen:
                continue
            seen.add(r["tic"])
            r["_why"] = label
            picked.append(r)
            took += 1
        print(f"[12] {label:28s} -> took {took} of {len(group)} available")

    print(f"\n[12] resolving TESS data products for {len(picked)} targets "
          f"(SPOC 120 s preferred)...\n")
    print(f"     {'planet':18s} {'TIC':>11} {'P[d]':>7} {'dur':>6} {'b':>5} {'e':>5} "
          f"{'Tmag':>5} {'sector':>7} {'cad':>6}")

    final = []
    for r in picked:
        try:
            search = search_products(r["tic"])
        except Exception as exc:
            rejected.append((r["pl_name"], f"product search failed ({type(exc).__name__})"))
            continue
        if len(search) == 0:
            rejected.append((r["pl_name"], "no TESS light curves"))
            continue

        tab = search.table
        authors = np.asarray(tab["author"], str)
        exps = np.asarray(tab["exptime"], float)
        secs = np.asarray(tab["sequence_number"], int)

        pick = None
        for au, cad in [("SPOC", 120.0), ("SPOC", 20.0), ("TESS-SPOC", None), ("QLP", None)]:
            m = (authors == au) if cad is None else ((authors == au) & (exps == cad))
            if m.any():
                i = int(np.flatnonzero(m)[0])
                pick = (au, float(exps[i]), int(secs[i]))
                break
        if pick is None:
            rejected.append((r["pl_name"], "no SPOC/TESS-SPOC/QLP product"))
            continue

        try:
            props = fetch_tic_properties(r["tic"])
            tmag = props.get("tmag")
        except Exception:
            tmag = None

        ratror = None
        # depth is not in the qualifying file; derive k from duration geometry later
        r.update(author=pick[0], exptime=pick[1], sector=pick[2], tmag=tmag, ratror=ratror)
        final.append(r)
        print(f"     {r['pl_name']:18s} {r['tic']:>11} {r['pl_orbper']:7.2f} "
              f"{r['pl_trandur']:6.2f} {r['pl_imppar']:5.2f} "
              f"{r['pl_orbeccen'] if r['pl_orbeccen'] is not None else float('nan'):5.2f} "
              f"{tmag if tmag else float('nan'):5.2f} {pick[2]:7d} {pick[1]:6.0f}")

    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pl_name", "tic", "sector", "author", "exptime", "pl_orbper",
                    "pl_trandur", "pl_imppar", "pl_orbeccen", "st_mass", "st_rad",
                    "st_teff", "tmag", "ratror", "why"])
        for r in final:
            w.writerow([r["pl_name"], r["tic"], r["sector"], r["author"], r["exptime"],
                        r["pl_orbper"], r["pl_trandur"], r["pl_imppar"], r["pl_orbeccen"],
                        r["st_mass"], r["st_rad"], r["st_teff"], r["tmag"], r["ratror"],
                        r["_why"]])

    print(f"\n[12] SELECTED {len(final)} targets -> {OUT}")
    if final:
        P = np.array([r["pl_orbper"] for r in final])
        e = np.array([r["pl_orbeccen"] for r in final if r["pl_orbeccen"] is not None])
        b = np.array([r["pl_imppar"] for r in final])
        tm = np.array([r["tmag"] for r in final if r["tmag"]])
        print(f"     period    {P.min():7.1f} - {P.max():7.1f} d   (median {np.median(P):.1f})")
        print(f"     ecc       {e.min():7.2f} - {e.max():7.2f}     (n={len(e)} with published e)")
        print(f"     b         {b.min():7.2f} - {b.max():7.2f}")
        print(f"     Tmag      {tm.min():7.2f} - {tm.max():7.2f}")
        cad = {}
        for r in final:
            cad[r["exptime"]] = cad.get(r["exptime"], 0) + 1
        print(f"     cadences  {dict(sorted(cad.items()))}")

    print(f"\n[12] REJECTED {len(rejected)}:")
    reasons = {}
    for name, why in rejected:
        reasons.setdefault(why, []).append(name)
    for why, names in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        print(f"     {len(names):3d}  {why}")
        if len(names) <= 6:
            print(f"          {', '.join(names)}")


if __name__ == "__main__":
    main()
