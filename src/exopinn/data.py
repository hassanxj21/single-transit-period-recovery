"""Real TESS data: download, detrend, locate the transit, extract features.

Two detrending problems this module fixes relative to the notebook pipeline:

1. ``lc.flatten(window_length=401)`` counts **cadences, not time**. On 2-minute
   SPOC data 401 cadences is 13.4 hours; on 30-minute FFI data it is 8.4 days.
   The same call therefore detrends on wildly different timescales depending on
   which product happened to download first. Here the window is specified in
   days and converted using the actual median cadence.

2. The Savitzky-Golay trend is fitted *through* the transit, so it absorbs part
   of the signal - severely for the 14-16 hour candidates, where the transit is
   a large fraction of a 13-hour window. Depth comes out shallow and duration
   short, and period scales as duration cubed. Here we detrend twice: once to
   locate the transit, then again with the transit cadences masked out of the
   trend fit.
"""

from __future__ import annotations

import warnings

import numpy as np

from .detrend import savgol_detrend
from .lightcurve import N_POINTS, WINDOW_DAYS, extract_window, measure_transit


# A MAST request with no timeout can hang indefinitely and is indistinguishable
# from slow progress. Bound every network call.
NETWORK_TIMEOUT = 120


def _set_timeouts():
    try:
        from astroquery.mast import Catalogs, Observations
        Observations.TIMEOUT = NETWORK_TIMEOUT
        Catalogs.TIMEOUT = NETWORK_TIMEOUT
    except Exception:
        pass


def fetch_tic_properties(tic: int) -> dict:
    """Stellar mass, radius and Teff from the TESS Input Catalog."""
    from astroquery.mast import Catalogs

    _set_timeouts()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat = Catalogs.query_criteria(catalog="TIC", ID=int(tic))
    if len(cat) == 0:
        raise LookupError(f"TIC {tic} not found")
    row = cat[0]

    def val(key):
        v = row[key]
        return None if v is None or np.ma.is_masked(v) or not np.isfinite(float(v)) else float(v)

    return {"tic": int(tic), "mass": val("mass"), "radius": val("rad"),
            "teff": val("Teff"), "tmag": val("Tmag")}


def search_products(tic: int, sector=None):
    import lightkurve as lk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lk.search_lightcurve(f"TIC {tic}", mission="TESS", sector=sector)


def download_lightcurve(tic: int, sector=None, author="SPOC", index=0):
    """Return (lc, provenance) for one TESS light curve, SPOC preferred."""
    search = search_products(tic, sector)
    if len(search) == 0:
        raise LookupError(f"no TESS light curves for TIC {tic} sector={sector}")

    pick = search
    if author is not None:
        try:
            sel = search[np.asarray(search.table["author"]) == author]
            if len(sel) > 0:
                pick = sel
        except Exception:
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc = pick[index].download()
    prov = {"author": str(pick.table["author"][index]),
            "sector": int(pick.table["sequence_number"][index]),
            "exptime": float(pick.table["exptime"][index]),
            "n_products": len(search)}
    return lc, prov


def _window_cadences(time, window_days):
    """Savitzky-Golay window length in cadences, from a window specified in days."""
    dt = float(np.median(np.diff(np.sort(time))))
    n = int(round(window_days / dt))
    return max(11, n | 1)  # odd, and long enough for the polynomial


def clean_and_flatten(lc, window_days=1.0, transit_mask=None):
    """Normalise and detrend, returning plain arrays (t, flat_flux, n_cadences).

    Uses ``detrend.savgol_detrend`` rather than ``lightkurve.flatten`` so that
    this path is identical to the one used to build the injection-recovery
    training set. Any behavioural difference between the two would reintroduce
    exactly the sim-to-real gap injection-recovery exists to remove.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = lc.remove_nans().normalize()
        t = np.asarray(clean.time.value, float)
        f = np.asarray(clean.flux.value, float)

    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    wl = _window_cadences(t, window_days)
    flat = savgol_detrend(t, f, window_days=window_days, transit_mask=transit_mask)
    return t, flat, wl


def find_transit(time, flux, search_window=None, min_depth_sigma=3.0, smooth_hours=0.5):
    """Locate the deepest dip. Returns (t0, depth, sigma_estimate) or None.

    ``search_window`` is an optional (t_start, t_end) bracket in the light
    curve's own time units, used when the transit has already been identified
    by eye.
    """
    t = np.asarray(time, float)
    f = np.asarray(flux, float)
    sel = np.ones(len(t), bool)
    if search_window is not None:
        sel = (t >= search_window[0]) & (t <= search_window[1])
    if sel.sum() < 10:
        return None

    dt = float(np.median(np.diff(np.sort(t))))
    w = max(3, int(round(smooth_hours / 24.0 / dt)) | 1)
    kern = np.ones(w) / w
    fs = np.convolve(np.pad(f[sel], w // 2, mode="edge"), kern, mode="same")[w // 2 : w // 2 + sel.sum()]

    # Robust scatter from the median absolute deviation of the unsmoothed flux.
    sigma = 1.4826 * np.median(np.abs(f - np.median(f)))
    i = int(np.argmin(fs))
    depth = float(np.median(fs) - fs[i])
    if depth < min_depth_sigma * sigma / np.sqrt(w):
        return None
    return float(t[sel][i]), depth, float(sigma)


# First-pass detrending window. Must be long compared with the transit: at 1 day
# an 11.4 h transit in NGTS-38 b was flattened to 3.05 h and 0.075% depth, which
# then sized the second-pass mask far too narrow. At 5 days the same unmasked
# pass recovers 11.45 h and 0.344%.
FIRST_PASS_DAYS = 5.0


def process_lightcurve(t, f, search_window=None, window_days=WINDOW_DAYS,
                       n_points=N_POINTS, n_iter=3, t0_hint=None):
    """Detrend, locate the transit, extract the window and measure it.

    Shared verbatim by real inference (``prepare_target``) and by the
    injection-recovery training-set builder, so the two paths cannot drift
    apart. Any difference between them would reintroduce the sim-to-real gap
    that injection-recovery exists to close.

    The mask width and filter window are iterated: each is sized from the
    current duration estimate, and a duration measured through an unprotected
    filter is always too short.

    Returns a dict, or None if no transit is found.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)

    flat = savgol_detrend(t, f, window_days=FIRST_PASS_DAYS)
    hit = find_transit(t, flat, search_window)
    if hit is None and t0_hint is None:
        return None
    t0 = t0_hint if t0_hint is not None else hit[0]

    grid = win = None
    depth = duration_hr = np.nan
    dur_days = 0.25

    span = float(t.max() - t.min()) if len(t) else 0.0
    for _ in range(max(1, n_iter)):
        guard = min(max(1.5 * dur_days, 0.15), 0.45 * window_days)
        # Cap the filter window relative to the available baseline. Savitzky-Golay
        # edge effects reach about half a filter window inward from each end, so
        # an uncapped window (8 x a 40 h duration = 14 d) would contaminate the
        # centre of a short segment while leaving a full sector untouched - a
        # length-dependent distortion, and therefore a domain gap between the
        # injected training set and real inference.
        detrend_days = max(FIRST_PASS_DAYS, 8.0 * dur_days)
        if span > 0:
            detrend_days = min(detrend_days, 0.30 * span)
        flat = savgol_detrend(t, f, window_days=detrend_days,
                              transit_mask=np.abs(t - t0) < guard)

        if t0_hint is None:
            again = find_transit(t, flat, search_window)
            if again is not None:
                t0 = again[0]

        grid, win = extract_window(t, flat, t0, window_days, n_points)
        if grid is None:
            return None
        win = win / np.median(win)
        depth, duration_hr, _ = measure_transit(grid, win)
        if not np.isfinite(duration_hr) or duration_hr <= 0:
            break
        new_dur_days = duration_hr / 24.0
        if abs(new_dur_days - dur_days) < 0.05 * dur_days:
            dur_days = new_dur_days
            break
        dur_days = new_dur_days

    sigma = 1.4826 * np.median(np.abs(win - np.median(win))) if win is not None else np.nan
    # Integrated transit SNR, the standard detection statistic and the same one
    # the synthetic/injected generators cut on. Per-cadence depth/sigma alone
    # understates a long shallow transit that is sampled many times.
    cadence_hr = window_days * 24.0 / n_points
    n_in = max(duration_hr / cadence_hr, 1.0) if np.isfinite(duration_hr) else 1.0
    snr_pt = float(depth / sigma) if sigma > 0 else np.nan
    return {"t_full": t, "flux_full": flat, "t0": float(t0), "t_grid": grid,
            "lc": np.asarray(win, np.float32), "depth": float(depth),
            "duration_hr": float(duration_hr), "scatter": float(sigma),
            "snr_point": snr_pt, "n_in_transit": float(n_in),
            "snr": snr_pt * np.sqrt(n_in) if np.isfinite(snr_pt) else np.nan}


BTJD_OFFSET = 2457000.0


def sector_with_transit(tic, tranmid_bjd, period_days, half_window=0.75, edge_margin=0.3,
                        prefer=(("SPOC", 120.0), ("SPOC", 20.0), ("TESS-SPOC", None),
                                ("QLP", None)), verbose=False, min_snr=7.5, max_tries=4):
    """Find a TESS sector that actually contains a transit, using the ephemeris.

    Long-period planets transit in only a small fraction of the sectors that
    observed them. Picking "the first available product" and searching it for
    the deepest dip therefore measures noise most of the time - which is exactly
    what happened before this existed (HD 56414 b: published 7.58 h duration,
    measured 1.82 h at SNR 1.6).

    SECTOR SELECTION RULE (explicit): minimise |n|, the number of cycles from the
    published epoch. Propagated timing uncertainty is sigma_t(n) = sqrt(sigma_T0^2
    + (n sigma_P)^2), which grows monotonically with |n|, so minimising |n| and
    minimising propagated uncertainty are the same rule. Photometric quality is
    NOT optimised: a later sector with better photometry is used only if the
    lowest-|n| sector fails the SNR gate. This is deliberate - a mis-predicted
    epoch measures noise, and no amount of photometric quality repairs that.

    Returns a dict with sector, window, t_transit, n_cycles, provenance, snr - or None.
    """
    t0_btjd = float(tranmid_bjd) - BTJD_OFFSET

    # Decide from MAST metadata, not by downloading. A CVZ target has 20+ sectors
    # and a long-period planet transits in almost none of them, so downloading
    # each one to read its time range costs ~9 minutes per target; the metadata
    # query costs ~4 seconds for all sectors at once.
    from astroquery.mast import Observations

    _set_timeouts()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obs = Observations.query_criteria(target_name=str(int(tic)),
                                              obs_collection="TESS",
                                              dataproduct_type="timeseries")
    except Exception:
        return None
    if obs is None or len(obs) == 0:
        return None

    # MJD -> BTJD: BTJD = JD - 2457000, MJD = JD - 2400000.5
    MJD_TO_BTJD = 2400000.5 - BTJD_OFFSET

    ranked = {}
    for row in obs:
        try:
            sec = int(row["sequence_number"])
            prov = str(row["provenance_name"]).upper()
            lo = float(row["t_min"]) + MJD_TO_BTJD
            hi = float(row["t_max"]) + MJD_TO_BTJD
        except (KeyError, ValueError, TypeError):
            continue
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        rank = next((i for i, (au, _) in enumerate(prefer) if au.upper() in prov), len(prefer))
        prev = ranked.get(sec)
        if prev is None or rank < prev[0]:
            ranked[sec] = (rank, lo, hi, prov)

    candidates = []
    for sec, (rank, lo, hi, prov) in ranked.items():
        a, b = lo + edge_margin, hi - edge_margin
        if b <= a:
            continue
        n0 = int(np.floor((a - t0_btjd) / period_days))
        for n in (n0, n0 + 1, n0 + 2):
            tt = t0_btjd + n * period_days
            if a <= tt <= b:
                # Rank by |n|, the number of cycles extrapolated from the published
                # epoch. Ephemeris uncertainty accumulates linearly with n, so a
                # far-extrapolated epoch can miss the transit entirely: TOI-2134 c
                # at n=0 (S52) gives depth 1.008% at SNR 182, while the same
                # ephemeris at n=8 (S80) gives 0.053% at SNR 2.5. Ranking by how
                # centrally the transit sits in its sector picked the latter.
                centrality = min(tt - lo, hi - tt)
                candidates.append((rank, abs(n), -centrality, sec, tt, prov))
                break

    if not candidates:
        return None
    candidates.sort()

    # Self-validating: take the first candidate whose extracted transit actually
    # clears min_snr, and otherwise fall back to the best one seen. Sector choice
    # is then decided by the data rather than by metadata alone.
    best = None
    for rank, ncyc, _, sector, tt, prov in candidates[:max_tries]:
        try:
            lc, _ = download_lightcurve(tic, sector=sector)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clean = lc.remove_nans().normalize()
                t = np.asarray(clean.time.value, float)
                f = np.asarray(clean.flux.value, float)
            ok = np.isfinite(t) & np.isfinite(f)
            t, f = t[ok], f[ok]
            if (np.abs(t - tt) < 0.25).sum() < 5:
                continue  # no coverage at the predicted epoch
            res = process_lightcurve(t, f, search_window=(tt - half_window, tt + half_window))
            snr = res["snr"] if res else 0.0
            if verbose:
                print(f"      S{sector} ({prov}) n={ncyc:+d} BTJD {tt:.3f} -> SNR {snr:.1f}")
            if best is None or snr > best[0]:
                best = (snr, sector, tt, prov, ncyc)
            if snr >= min_snr:
                break
        except Exception:
            continue

    if best is None:
        return None
    snr, sector, tt, prov, ncyc = best
    if verbose:
        print(f"      chose S{sector} ({prov}), {ncyc} cycles from published epoch, SNR {snr:.1f}")
    return {"sector": sector, "window": (tt - half_window, tt + half_window),
            "t_transit": tt, "n_cycles": ncyc, "provenance": prov, "snr": snr}


def prepare_target(tic, sector=None, search_window=None, window_days=WINDOW_DAYS,
                   n_points=N_POINTS, detrend_days=1.0, mass=None, radius=None, verbose=True):
    """Full real-data path: download -> detrend (transit-masked) -> window -> measure.

    Returns a dict with the network's two inputs (``lc``, ``scalars``) plus
    everything needed to audit the measurement.
    """
    lc, prov = download_lightcurve(tic, sector)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = lc.remove_nans().normalize()
        t = np.asarray(clean.time.value, float)
        f = np.asarray(clean.flux.value, float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]

    res = process_lightcurve(t, f, search_window=search_window,
                             window_days=window_days, n_points=n_points)
    if res is None:
        return {"tic": tic, "provenance": prov, "error": "no transit found"}

    t0, grid, win = res["t0"], res["t_grid"], res["lc"]
    depth, duration_hr, sigma = res["depth"], res["duration_hr"], res["scatter"]

    props = {"mass": mass, "radius": radius}
    if mass is None or radius is None:
        tic_props = fetch_tic_properties(tic)
        props["mass"] = mass if mass is not None else tic_props["mass"]
        props["radius"] = radius if radius is not None else tic_props["radius"]
        props["teff"] = tic_props["teff"]

    out = {
        "tic": tic, "provenance": prov,
        "t_full": res["t_full"], "flux_full": res["flux_full"], "t0": t0,
        "t_grid": grid, "lc": np.asarray(win, np.float32),
        "depth": depth, "duration_hr": duration_hr,
        "scatter": sigma, "snr": res["snr"],
        "snr_point": res.get("snr_point"), "n_in_transit": res.get("n_in_transit"),
        **props,
    }
    if props["mass"] and props["radius"]:
        out["scalars"] = np.array([[duration_hr, depth, props["mass"], props["radius"]]], np.float32)

    if verbose:
        print(f"  TIC {tic}: {prov['author']} S{prov['sector']} {prov['exptime']:.0f}s")
        print(f"    t0={t0:.4f}  depth={depth*100:.3f}%  duration(FWHM)={duration_hr:.2f}h  "
              f"SNR={out['snr']:.1f}")
    return out
