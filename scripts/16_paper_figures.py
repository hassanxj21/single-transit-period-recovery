#!/usr/bin/env python
"""Generate every paper figure into paper/figures/ as 300 dpi PNG.

Colourblind-safe throughout (Okabe-Ito). Flat directory, exact filenames.

    python scripts/16_paper_figures.py
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from exopinn import data as D                                        # noqa: E402
from exopinn.bls import run_bls                                      # noqa: E402
from exopinn.lightcurve import simulate_transit                      # noqa: E402
from exopinn.physics import (equilibrium_temperature,                # noqa: E402
                             period_from_duration, transit_duration)
from exopinn.targets import CANDIDATES, CONFIRMED, EXPANDED          # noqa: E402
from exopinn.train import load_model, predict                        # noqa: E402

FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "results" / "data" / "_figcache.pkl"
RESULTS = ROOT / "results" / "data" / "expanded_results.json"

# Okabe-Ito colourblind-safe palette
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7"}
COL = {"bls27": OI["verm"], "bls200": OI["orange"], "analytic": OI["blue"],
       "scalars": OI["green"], "synth full": OI["purple"], "inj full": OI["sky"],
       "inj scalars": OI["black"]}
MARK = {"bls27": "s", "bls200": "D", "analytic": "^", "scalars": "o",
        "synth full": "v", "inj full": "P", "inj scalars": "X"}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 9.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 0.9,
    "lines.linewidth": 1.6, "figure.constrained_layout.use": False,
})

WRITTEN, SKIPPED = [], []


def save(fig, name):
    p = FIGDIR / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    WRITTEN.append(name)
    print(f"  wrote {name}", flush=True)


def skip(name, why):
    SKIPPED.append((name, why))
    print(f"  SKIP  {name}: {why}", flush=True)


# ---------------------------------------------------------------- data loading
payload = json.loads(RESULTS.read_text())
recs = payload["records"]
min_snr = payload.get("min_snr", 7.5)
HEAD = [r for r in recs if r["snr"] >= min_snr and 3.0 <= r["true_period"] <= 1000.0]
METHODS = [m for m in payload["methods"]]
MODEL_METHODS = [m for m in METHODS if m not in ("bls27", "bls200")]
BY_LABEL = {r["label"]: r for r in recs}

_cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}


def prep(t):
    key = (t.tic, t.sector, getattr(t, "tranmid", None))
    if key in _cache:
        return _cache[key]
    sector, window = t.sector, t.window
    if getattr(t, "tranmid", None) and t.true_period:
        hit = D.sector_with_transit(t.tic, t.tranmid, t.true_period)
        if hit is None:
            _cache[key] = None
            CACHE.write_bytes(pickle.dumps(_cache))
            return None
        sector, window = hit["sector"], hit["window"]
    try:
        p = D.prepare_target(t.tic, sector=sector, search_window=window,
                             mass=t.mass, radius=t.radius, verbose=False)
    except Exception:
        p = None
    if p is not None and "error" in p:
        p = None
    _cache[key] = p
    CACHE.write_bytes(pickle.dumps(_cache))
    return p


ALLT = {t.label: t for t in CONFIRMED + EXPANDED}


def err(v, tr):
    return abs(v - tr) / tr * 100 if (np.isfinite(v) and tr) else np.nan


def signed(v, tr):
    return (v - tr) / tr * 100 if (np.isfinite(v) and tr) else np.nan


# =============================================================== fig01
def fig01():
    want = ["NGTS-38 b", "TOI-2134 c", "TOI-199 b"]
    cols = []
    for lab in want:
        t = ALLT.get(lab)
        if t is None:
            continue
        p = prep(t)
        if p is None or "scalars" not in p:
            continue
        cols.append((t, p))
    if not cols:
        return skip("fig01_lightcurves.png", "no confirmed target could be prepared")

    fig, axes = plt.subplots(2, len(cols), figsize=(4.6 * len(cols), 6.4), squeeze=False)
    for j, (t, p) in enumerate(cols):
        ax = axes[0][j]
        ax.plot(p["t_full"], p["flux_full"], ".", ms=0.8, alpha=0.45, color=OI["black"],
                rasterized=True)
        ax.axvline(p["t0"], color=OI["verm"], lw=1.4)
        ax.set_title(f"{t.label}\nsector {p['provenance']['sector']}, "
                     f"{p['provenance']['exptime']:.0f} s", fontsize=11)
        ax.set_xlabel("BTJD [days]")
        if j == 0:
            ax.set_ylabel("normalised flux")
        lo, hi = np.percentile(p["flux_full"], [0.3, 99.7])
        pad = 0.25 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

        ax = axes[1][j]
        hrs = (p["t_grid"] - p["t0"]) * 24.0
        ax.plot(hrs, p["lc"], "o", ms=2.6, color=OI["black"], alpha=0.75, label="data")
        k = np.sqrt(max(p["depth"], 0.0))
        b = t.known_b if t.known_b is not None else 0.0
        e = t.known_ecc if t.known_ecc is not None else 0.0
        if t.true_period:
            model = simulate_transit(p["t_grid"], p["t0"], t.true_period, p["mass"],
                                     p["radius"], k, b, e, -np.pi / 2, 0.4, 0.25)
            ax.plot(hrs, model, "-", color=OI["blue"], lw=1.8,
                    label=f"model (P={t.true_period:.1f} d, b={b:.2f}, e={e:.2f})")
        ax.axhline(1.0, color="grey", lw=0.7, ls=":")
        ax.set_xlabel("hours from mid-transit")
        if j == 0:
            ax.set_ylabel("normalised flux")
        ax.annotate(f"depth = {p['depth']*100:.3f}%\nFWHM = {p['duration_hr']:.2f} h\n"
                    f"SNR = {p['snr']:.1f}",
                    xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="grey", alpha=0.85))
        ax.legend(loc="upper right", fontsize=8)
    save(fig, "fig01_lightcurves.png")


# =============================================================== fig02
def fig02():
    t = ALLT.get("NGTS-38 b")
    p = prep(t) if t else None
    if p is None:
        return skip("fig02_bls_periodogram_flat.png", "NGTS-38 b could not be prepared")
    r = run_bls(p["t_full"], p["flux_full"], 0.5, 400.0, n_periods=20000)
    grid, power = r["grid"], r["power_spectrum"]

    # Where does the spectrum stop carrying information? Beyond the longest
    # contiguous data chunk every trial period predicts exactly one transit in
    # the window, so BLS returns a numerically identical power.
    d = np.abs(np.diff(power))
    nz = np.flatnonzero(d > 1e-12)
    p_flat = float(grid[nz[-1]]) if len(nz) else float(grid[0])
    beyond = power[grid > 27.0]
    n_beyond, n_uniq = int((grid > 27.0).sum()), int(len(np.unique(np.round(beyond, 12))))
    frac_flat = float((grid > p_flat).mean())

    # SDE is undefined over the whole grid (MAD = 0). Quote it over the
    # informative region only, and say so.
    info = power[grid <= p_flat]
    med, mad = np.median(info), 1.4826 * np.median(np.abs(info - np.median(info)))
    sde_info = (np.max(info) - med) / mad if mad > 0 else np.nan

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.axvspan(p_flat, grid.max(), color=OI["yellow"], alpha=0.22, zorder=0,
               label=f"power numerically constant (> {p_flat:.1f} d)")
    ax.plot(grid, power, lw=0.8, color=OI["black"], zorder=3)
    ax.axvline(27.0, color=OI["orange"], ls="--", lw=1.7, zorder=4,
               label="27 d nominal sector baseline")
    ax.axvline(180.53, color=OI["verm"], lw=2.2, zorder=4,
               label="true period 180.53 d")
    ax.set_xscale("log")
    ax.set_xlabel("trial period [days]")
    ax.set_ylabel("BLS power")
    ax.set_title("NGTS-38 b: the BLS periodogram carries no information beyond the baseline")
    ax.annotate(
        f"beyond 27 d: {n_uniq} distinct power value across {n_beyond:,} trial periods\n"
        f"(range identically zero; {frac_flat:.0%} of the grid is bit-identical)\n"
        f"SDE undefined over the full grid — spectrum MAD = 0\n"
        f"SDE within the informative region (< {p_flat:.1f} d) = {sde_info:.1f}, "
        f"vs SDE > 7 for a detection",
        xy=(0.025, 0.62), xycoords="axes fraction", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.92))
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "fig02_bls_periodogram_flat.png")


# =============================================================== fig03
def fig03():
    uppers = [27.0, 50.0, 75.0, 100.0, 150.0, 200.0]
    meds, n_used = [], 0
    per_upper = {u: [] for u in uppers}
    for r in HEAD:
        t = ALLT.get(r["label"])
        if t is None:
            continue
        p = prep(t)
        if p is None:
            continue
        n_used += 1
        for u in uppers:
            try:
                res = run_bls(p["t_full"], p["flux_full"], 0.5, u, n_periods=8000)
                per_upper[u].append(err(res["period"], r["true_period"]))
            except Exception:
                pass
    if n_used < 3:
        return skip("fig03_bls_grid_insensitivity.png",
                    f"only {n_used} targets could be reprocessed")
    meds = [np.median(per_upper[u]) for u in uppers]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(uppers, meds, "o-", color=OI["verm"], ms=8, label=f"BLS median error (n={n_used})")
    for u in uppers:
        v = per_upper[u]
        ax.plot([u] * len(v), v, ".", ms=3, color=OI["orange"], alpha=0.45)
    ax.axhline(np.median(meds), ls=":", color="grey", lw=1)
    ax.set_xlabel("BLS grid upper bound [days]")
    ax.set_ylabel("median |period error| [%]")
    ax.set_ylim(0, 105)
    ax.set_title("Widening the BLS search grid does not help")
    ax.annotate(f"median error {min(meds):.0f}-{max(meds):.0f}% across a 7x range\n"
                f"in grid upper bound: the failure is structural,\nnot a grid-width choice",
                xy=(0.30, 0.12), xycoords="axes fraction", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))
    ax.legend(loc="upper right")
    save(fig, "fig03_bls_grid_insensitivity.png")


# =============================================================== fig04
def fig04():
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    tr = np.array([r["true_period"] for r in HEAD])
    lim = [tr.min() * 0.35, tr.max() * 3.0]
    for m in METHODS:
        v = np.array([r.get(m, np.nan) for r in HEAD])
        ok = np.isfinite(v) & (v > 0)
        ax.scatter(tr[ok], v[ok], s=46, alpha=0.85, label=m, marker=MARK.get(m, "o"),
                   color=COL.get(m, "grey"), edgecolor="white", linewidth=0.5)
    ax.plot(lim, lim, "-", color="grey", lw=1.2, zorder=0)
    for f in (2.0, 0.5):
        ax.plot(lim, [f * x for x in lim], ":", color="grey", lw=0.8, zorder=0)
    ax.axhspan(lim[0], 27.0, color=OI["yellow"], alpha=0.18, zorder=-1)
    ax.text(lim[0] * 1.15, 21, "reachable by a 27 d BLS grid", fontsize=8.5, color="#7a6a00")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true period [days]")
    ax.set_ylabel("predicted period [days]")
    ax.set_title(f"Single-transit period recovery (n = {len(HEAD)})")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(alpha=0.22, which="both")
    save(fig, "fig04_predicted_vs_true.png")


# =============================================================== fig05
def fig05():
    labs, vals, cols = [], [], []
    for m in METHODS:
        e = [err(r.get(m, np.nan), r["true_period"]) for r in HEAD
             if np.isfinite(r.get(m, np.nan)) and r.get(m, 0) > 0]
        if e:
            labs.append(m); vals.append(np.median(e)); cols.append(COL.get(m, "grey"))
    order = np.argsort(vals)[::-1]
    labs = [labs[i] for i in order]; vals = [vals[i] for i in order]
    cols = [cols[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.barh(range(len(vals)), vals, color=cols, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=10)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    ax.set_xlabel("median |period error| [%]")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title(f"Median absolute error (n = {len(HEAD)}), lower is better")
    ax.grid(alpha=0.25, axis="x")
    save(fig, "fig05_median_error.png")


# =============================================================== fig06
def fig06():
    wb = [r for r in HEAD if r.get("known_b") is not None and np.isfinite(r.get("known_b", np.nan))]
    if len(wb) < 5:
        return skip("fig06_error_vs_b.png", "fewer than 5 targets with published b")
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    lines = []
    for m in MODEL_METHODS:
        bs = [r["known_b"] for r in wb if np.isfinite(r.get(m, np.nan))]
        es = [err(r[m], r["true_period"]) for r in wb if np.isfinite(r.get(m, np.nan))]
        if len(bs) < 5:
            continue
        rho, pv = spearmanr(bs, es)
        ax.scatter(bs, es, s=48, alpha=0.85, marker=MARK.get(m, "o"),
                   color=COL.get(m, "grey"), edgecolor="white", linewidth=0.5,
                   label=f"{m}   $\\rho$={rho:+.2f}, p={pv:.3f}")
        lines.append(m)
    ax.axvline(1.0, ls=":", color="grey", lw=1)
    ax.set_xlabel("published impact parameter $b$")
    ax.set_ylabel("|period error| [%]")
    ax.set_title(f"Error against impact parameter (n = {len(wb)})")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.25)
    save(fig, "fig06_error_vs_b.png")


# =============================================================== fig07
def fig07():
    wb = [r for r in HEAD if r.get("known_b") is not None and np.isfinite(r.get("known_b", np.nan))]
    if len(wb) < 5:
        return skip("fig07_signed_error_mechanism.png", "fewer than 5 targets with published b")
    bs, s_unc, s_trueb, s_scal = [], [], [], []
    for r in wb:
        k = np.sqrt(max(r["depth"], 0.0))
        p_tb = period_from_duration(r["duration_hr"], r["mass"], r["radius"],
                                    k=k, b=r["known_b"], contact="fwhm")
        bs.append(r["known_b"])
        s_unc.append(signed(r["analytic"], r["true_period"]))
        s_trueb.append(signed(p_tb, r["true_period"]))
        s_scal.append(signed(r.get("scalars", np.nan), r["true_period"]))

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for ys, lab, c, mk in [
            (s_unc, f"analytic, b=0 (median {np.nanmedian(s_unc):+.0f}%)", OI["blue"], "^"),
            (s_trueb, f"analytic, true b (median {np.nanmedian(s_trueb):+.0f}%)", OI["sky"], "P"),
            (s_scal, f"scalars model (median {np.nanmedian(s_scal):+.0f}%)", OI["green"], "o")]:
        ax.scatter(bs, ys, s=52, alpha=0.85, color=c, marker=mk,
                   edgecolor="white", linewidth=0.5, label=lab)
        ax.axhline(np.nanmedian(ys), color=c, ls="--", lw=1.1, alpha=0.65)
    ax.axhline(0, color=OI["black"], lw=1.4)
    ax.set_xlabel("published impact parameter $b$")
    ax.set_ylabel("signed period error  (pred - true)/true  [%]")
    ax.set_title("Mechanism: the b=0 inversion underestimates almost universally")
    ax.annotate("14/15 below truth for b=0;\ncorrecting b recovers only ~20 pp;\n"
                "residual is eccentricity (see fig08)",
                xy=(0.03, 0.05), xycoords="axes fraction", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    save(fig, "fig07_signed_error_mechanism.png")


# =============================================================== fig08
def fig08():
    we = [r for r in HEAD if r.get("known_ecc") is not None
          and np.isfinite(r.get("known_ecc", np.nan))]
    if len(we) < 5:
        return skip("fig08_error_vs_eccentricity.png", "fewer than 5 targets with published e")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    for m in MODEL_METHODS:
        es = [r["known_ecc"] for r in we if np.isfinite(r.get(m, np.nan))]
        vs = [err(r[m], r["true_period"]) for r in we if np.isfinite(r.get(m, np.nan))]
        if len(es) < 5:
            continue
        rho, pv = spearmanr(es, vs)
        ax.scatter(es, vs, s=48, alpha=0.85, marker=MARK.get(m, "o"),
                   color=COL.get(m, "grey"), edgecolor="white", linewidth=0.5,
                   label=f"{m}   $\\rho$={rho:+.2f}, p={pv:.3f}")
    ax.set_xlabel("published eccentricity $e$")
    ax.set_ylabel("|period error| [%]")
    ax.set_title("Error against eccentricity")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.25)

    # residual duration deficit
    ee, qq, labs = [], [], []
    for r in we:
        if r.get("known_b") is None or not np.isfinite(r.get("known_b", np.nan)):
            continue
        k = np.sqrt(max(r["depth"], 0.0))
        mdl = transit_duration(r["true_period"], r["mass"], r["radius"],
                               k=k, b=r["known_b"], contact="fwhm")
        if not np.isfinite(mdl) or mdl <= 0:
            continue
        ee.append(r["known_ecc"]); qq.append(r["duration_hr"] / float(mdl))
        labs.append(r["label"])
    if len(ee) >= 5:
        rho, pv = spearmanr(ee, qq)
        ax2.scatter(ee, qq, s=64, color=OI["purple"], edgecolor="white", linewidth=0.6)
        ax2.axhline(1.0, color=OI["black"], lw=1.3)
        z = np.polyfit(ee, qq, 1)
        xs = np.linspace(min(ee), max(ee), 50)
        ax2.plot(xs, np.polyval(z, xs), "--", color=OI["purple"], lw=1.4)
        for e_, q_, l_ in zip(ee, qq, labs):
            if q_ > 1.05 or q_ < 0.5:
                ax2.annotate(l_, (e_, q_), fontsize=7.5, xytext=(4, 3),
                             textcoords="offset points")
        ax2.set_xlabel("published eccentricity $e$")
        ax2.set_ylabel("observed / circular-model duration")
        ax2.set_title(f"Residual duration deficit after correcting $b$\n"
                      f"$\\rho$ = {rho:+.3f}, p = {pv:.3f}, n = {len(ee)}")
        ax2.grid(alpha=0.25)
    save(fig, "fig08_error_vs_eccentricity.png")


# ---------------------------------------------------- NGTS-38 b posterior (9-11)
ECC_PRIOR = (0.867, 3.03)


def ngts_posterior(n_mc=20000, seed=7):
    t = ALLT.get("NGTS-38 b")
    k, M, R, T14, teff = t.known_k, t.mass, t.radius, t.duration_hr, t.teff
    rng = np.random.default_rng(seed)
    b = rng.uniform(0.0, 1.0 + k, n_mc)
    e = np.minimum(beta_dist.rvs(*ECC_PRIOR, size=n_mc, random_state=rng), 0.9)
    om = rng.uniform(0.0, 2 * np.pi, n_mc)
    out = np.empty(n_mc)
    for i in range(n_mc):
        out[i] = period_from_duration(T14, M, R, k=k, b=b[i], ecc=e[i],
                                      omega=om[i], contact="14")
    good = out[np.isfinite(out) & (out > 0)]
    return t, good, b, e


def fig09_10_11():
    if "NGTS-38 b" not in ALLT:
        for n in ("fig09_ngts38b_posterior.png", "fig10_ngts38b_degeneracy.png",
                  "fig11_ngts38b_temperature.png"):
            skip(n, "NGTS-38 b not in target list")
        return
    t, post, _, _ = ngts_posterior()
    q = np.percentile(post, [2.5, 16, 50, 84, 97.5])
    pct = 100 * np.mean(post < t.true_period)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.hist(np.log10(post), bins=90, density=True, color=OI["sky"],
            edgecolor="white", linewidth=0.2)
    ax.axvspan(np.log10(q[1]), np.log10(q[3]), color=OI["black"], alpha=0.13,
               label=f"68%: {q[1]:.0f}-{q[3]:.0f} d")
    ax.axvline(np.log10(t.true_period), color=OI["verm"], lw=2.4,
               label=f"truth {t.true_period:.1f} d")
    ax.axvline(np.log10(27.0), color=OI["orange"], ls="--", lw=1.6,
               label="27 d sector baseline")
    ax.axvline(np.log10(q[2]), color=OI["blue"], ls=":", lw=1.8,
               label=f"posterior median {q[2]:.0f} d")
    ax.set_xlabel(r"$\log_{10}$ period [days]")
    ax.set_ylabel("posterior density")
    ax.set_title("NGTS-38 b: physics-only posterior from a single transit")
    ax.annotate(f"truth at the {pct:.1f}th percentile\nno period search range used",
                xy=(0.62, 0.62), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "fig09_ngts38b_posterior.png")

    # fig10 degeneracy plane
    k = t.known_k
    bs = np.linspace(0.0, 1.0 + k - 1e-3, 200)
    es = np.linspace(0.0, 0.8, 200)
    Z = np.array([[period_from_duration(t.duration_hr, t.mass, t.radius, k=k, b=bb,
                                        ecc=ee, omega=-np.pi / 2, contact="14")
                   for bb in bs] for ee in es])
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    im = ax.pcolormesh(bs, es, np.log10(Z), cmap="viridis", shading="auto")
    cs = ax.contour(bs, es, np.log10(Z),
                    levels=np.log10([30, 60, 120, t.true_period, 400, 800]),
                    colors="white", linewidths=0.9)
    ax.clabel(cs, fmt=lambda v: f"{10**v:.0f} d", fontsize=8)
    ct = ax.contour(bs, es, np.log10(Z), levels=[np.log10(t.true_period)],
                    colors=[OI["verm"]], linewidths=2.6)
    ax.plot(t.known_b, t.known_ecc, "*", ms=20, color=OI["yellow"],
            markeredgecolor=OI["black"], markeredgewidth=1.0,
            label=f"published (b={t.known_b:.2f}, e={t.known_ecc:.2f})")
    ax.set_xlabel("impact parameter $b$")
    ax.set_ylabel("eccentricity $e$")
    ax.set_title("NGTS-38 b: period implied by a fixed 14.86 h transit\n"
                 r"($\omega = -90^\circ$; red contour = true period)")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ implied period [days]")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig10_ngts38b_degeneracy.png")

    # fig11 temperature
    teq = equilibrium_temperature(t.teff, t.radius, post, t.mass)
    tq = np.percentile(teq, [16, 50, 84])
    hz = np.mean((teq >= 200) & (teq <= 320))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.hist(teq, bins=90, density=True, color=OI["orange"],
            edgecolor="white", linewidth=0.2)
    ax.axvspan(200, 320, color=OI["green"], alpha=0.22, label="habitable zone 200-320 K")
    ax.axvspan(tq[0], tq[2], color=OI["black"], alpha=0.13,
               label=f"68%: {tq[0]:.0f}-{tq[2]:.0f} K")
    ax.axvline(equilibrium_temperature(t.teff, t.radius, t.true_period, t.mass),
               color=OI["verm"], lw=2.4, label="truth")
    ax.set_xlim(0, min(1400, np.percentile(teq, 99)))
    ax.set_xlabel("equilibrium temperature [K]")
    ax.set_ylabel("posterior density")
    ax.set_title("NGTS-38 b: temperature is far better constrained than period")
    ax.annotate(f"P(habitable zone) = {hz:.1%}\n"
                r"$T_{eq}\propto P^{-1/3}$, so a factor-"
                f"{q[3]/q[1]:.1f} period interval\ncompresses to factor-{tq[2]/tq[0]:.1f}",
                xy=(0.42, 0.60), xycoords="axes fraction", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "fig11_ngts38b_temperature.png")


# =============================================================== fig12
def fig12():
    dirs = [("scalars", "cnn_pinn_scalars_only"), ("synth full", "cnn_pinn_full"),
            ("inj full", "cnn_pinn_inj_full"), ("inj scalars", "cnn_pinn_inj_scalars")]
    have = []
    for lab, d in dirs:
        h = ROOT / "results" / "models" / d / "history.json"
        if h.exists():
            have.append((lab, json.loads(h.read_text())))
    if not have:
        return skip("fig12_training_loss.png", "no history.json found")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for lab, h in have:
        c = COL.get(lab, "grey")
        ax.plot(h["epoch"], h["train_data"], "-", color=c, lw=1.3, alpha=0.55)
        ax.plot(h["epoch"], h["val_nll"], "-", color=c, lw=2.0, label=f"{lab} (val)")
        v = np.asarray(h["val_nll"], float)
        i = int(np.argmin(v))
        ax.plot(i, v[i], "o", color=c, ms=9, markeredgecolor="white", markeredgewidth=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Gaussian NLL")
    ax.set_title("Training: thin = train, thick = validation, marker = early-stopping point")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    save(fig, "fig12_training_loss.png")


# =============================================================== fig13
def fig13():
    try:
        model, norm, _ = load_model(ROOT / "results" / "models" /
                                    "cnn_pinn_inj_scalars" / "cnn_pinn.pt")
    except Exception as exc:
        return skip("fig13_candidates.png", f"model unavailable ({type(exc).__name__})")
    panels = []
    for t in CANDIDATES:
        p = prep(t)
        if p is None or "scalars" not in p:
            continue
        pr, lo, hi = predict(model, norm, p["lc"][None, :], p["scalars"])
        panels.append((t, p, float(pr[0]), float(lo[0]), float(hi[0])))
    if not panels:
        return skip("fig13_candidates.png", "no candidate could be prepared")

    n = len(panels)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.5 * nrow), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for i, (t, p, pr, lo, hi) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        mu, sig = np.log10(pr), (np.log10(hi) - np.log10(lo)) / 2.0
        xs = np.linspace(mu - 4 * sig, mu + 4 * sig, 400)
        ax.plot(10**xs, np.exp(-0.5 * ((xs - mu) / sig) ** 2), color=OI["green"], lw=2.0)
        ax.fill_between(10**xs, 0, np.exp(-0.5 * ((xs - mu) / sig) ** 2),
                        color=OI["green"], alpha=0.28)
        ax.axvline(pr, color=OI["black"], lw=1.6)
        ax.axvspan(lo, hi, color=OI["black"], alpha=0.12)
        ax.axvline(27.0, color=OI["orange"], ls="--", lw=1.4)
        ax.set_xscale("log")
        ax.set_yticks([])
        ax.set_xlabel("period [days]")
        ax.set_title(f"TIC {t.tic}", fontsize=11)
        ax.annotate(f"{pr:.0f} d\n[{lo:.0f}, {hi:.0f}]\n"
                    f"depth {p['depth']*100:.3f}%\nFWHM {p['duration_hr']:.2f} h",
                    xy=(0.03, 0.55), xycoords="axes fraction", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))
    fig.suptitle("Unconfirmed single-transit candidates: period posteriors "
                 "(orange dashed = 27 d sector baseline)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig13_candidates.png")


import os
_only = os.environ.get('ONLY')
_all = [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09_10_11, fig12, fig13]
for fn in ([f for f in _all if f.__name__ in _only.split(',')] if _only else _all):
    try:
        fn()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        skip(getattr(fn, "__name__", "?"), f"{type(exc).__name__}: {exc}")

print("\n" + "=" * 78)
print(f"DIRECTORY: {FIGDIR}")
print("=" * 78)
EXPECTED = ["fig01_lightcurves.png", "fig02_bls_periodogram_flat.png",
            "fig03_bls_grid_insensitivity.png", "fig04_predicted_vs_true.png",
            "fig05_median_error.png", "fig06_error_vs_b.png",
            "fig07_signed_error_mechanism.png", "fig08_error_vs_eccentricity.png",
            "fig09_ngts38b_posterior.png", "fig10_ngts38b_degeneracy.png",
            "fig11_ngts38b_temperature.png", "fig12_training_loss.png",
            "fig13_candidates.png"]
for name in EXPECTED:
    p = FIGDIR / name
    if p.exists():
        print(f"  OK      {name:38s} {p.stat().st_size // 1024:5d} KB")
    else:
        why = next((w for n, w in SKIPPED if n == name), "not generated")
        print(f"  MISSING {name:38s} {why}")
print(f"\n{sum(1 for n in EXPECTED if (FIGDIR / n).exists())}/{len(EXPECTED)} figures present")
