"""Target list: confirmed validation planets and unconfirmed single-transit candidates.

Stellar properties from the TESS Input Catalog. Durations flagged `reliable=False`
were measured as exactly the width of the inspection window, i.e. the half-depth
crossings were never found - those numbers are window artefacts, not transits,
and must be re-measured before they are used for anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Target:
    tic: int
    sector: int | None
    mass: float | None = None      # M_sun
    radius: float | None = None    # R_sun
    teff: float | None = None      # K
    depth: float | None = None     # fractional
    duration_hr: float | None = None
    # Which duration the number above actually is. Literature values and
    # ingress-to-egress eyeball brackets are T14; this pipeline's own
    # measure_transit() returns the full width at half depth. Mixing the two
    # shifts the implied period by ~30%, because period scales as duration^3.
    duration_kind: str = "14"
    true_period: float | None = None   # days, if confirmed
    name: str = ""
    window: tuple[float, float] | None = None  # BTJD bracket containing the transit
    duration_reliable: bool = True
    notes: str = ""
    # Published geometry, where a follow-up solution exists. Never fed to the
    # model - used only to check how far the b/e degeneracy moves the answer.
    known_b: float | None = None
    known_ecc: float | None = None
    known_k: float | None = None
    tranmid: float | None = None   # published mid-transit time, BJD
    period_err: float | None = None
    tranmid_err: float | None = None

    @property
    def label(self):
        return self.name or f"TIC {self.tic}"


# Usable baseline per TESS sector (27.4 d orbit pair minus the downlink gap).
# A planet is guaranteed to show at most one transit in a sector only when its
# period exceeds this; below it, seeing one transit rather than two is a phase
# accident, not a property of the system.
USABLE_BASELINE = 25.4

# PHYSICS-VALIDATION ONLY - not single-transit targets.
# TIC 393818343 has P = 16.25 d against a 25.4 d usable baseline, so it completes
# ~1.6 orbits per sector and would ordinarily show TWO transits. It is retained
# because its known period tests the forward model and the extraction pipeline,
# but it must not appear in any single-transit claim or in the headline medians.
VALIDATION_ONLY = [
    Target(tic=393818343, sector=55, mass=1.000, radius=1.100, teff=5657,
           depth=0.0131, duration_hr=5.5, true_period=16.25,
           notes="NOT single-transit: P=16.25 d < 25.4 d usable baseline (~1.6 transits "
                 "per sector). Physics/pipeline validation only."),
]

# Confirmed single-transit planets used to validate period recovery.
CONFIRMED = [
    Target(tic=172370679, sector=15, mass=0.602, radius=0.618, teff=3854,
           depth=0.0445, duration_hr=6.5, true_period=29.09, name="TOI-1899",
           notes="Observed duration 1.31x the circular-model value even with (1+k); "
                 "impact parameter can only shorten a transit, so this requires eccentricity."),
    # NASA Exoplanet Archive, pscomppars, queried 2026-07-26. Depth is (Rp/R*)^2
    # from the published ratror; the archive's pl_trandep is null for this planet.
    # Sector 33 (TESS-SPOC, 600 s) is the one holding the transit: a 0.335% dip at
    # BTJD 2209.17, 7.9x the 426 ppm scatter, matching the k=0.0591 expectation of
    # 0.35%. Sectors 7, 61, 87 and 88 show nothing above noise - as a genuine
    # single-transit planet should.
    Target(tic=65910228, sector=33, window=(2208.4, 2210.0),
           mass=1.460, radius=1.880, teff=6310,
           depth=0.0591**2, duration_hr=14.86, true_period=180.52797, name="NGTS-38 b",
           known_b=0.8536, known_ecc=0.3086, known_k=0.0591,
           notes="The decisive case. 180.5 d exceeds the 27 d sector baseline, so BLS cannot "
                 "recover it at any search range. Published b=0.854 and e=0.309 pull the "
                 "duration in opposite directions and nearly cancel: assuming b=0/e=0 gives "
                 "101 d (0.56x), assuming b=0.85/e=0 gives 478 d (2.65x), and the true "
                 "geometry reproduces 188 d (1.04x). The b-e spread at fixed observables is "
                 "a factor of ~4.7 - that spread IS the irreducible uncertainty."),
]

# Unconfirmed single-transit candidates: period unknown, this is the target class.
CANDIDATES = [
    Target(tic=233577004, sector=14, mass=1.040, radius=1.473, teff=5810,
           depth=0.0056, duration_hr=8.0, window=(1703.5, 1705.5), duration_reliable=False,
           notes="Reported 24.0 h = inspection window width. Re-measure."),
    Target(tic=341687821, sector=21, mass=1.192, radius=1.350, teff=6204,
           depth=0.0018, duration_hr=14.4, window=(1889.5, 1891.5)),
    Target(tic=122522333, sector=3, mass=1.200, radius=1.763, teff=6230,
           depth=0.0024, duration_hr=14.4, window=(1388.5, 1390.5)),
    Target(tic=438122862, sector=6, mass=1.650, radius=2.031, teff=7298,
           depth=0.0022, duration_hr=6.0, window=(1475.0, 1477.0), duration_reliable=False,
           notes="Reported 24.0 h = inspection window width. Re-measure."),
    Target(tic=232616346, sector=20, mass=0.950, radius=1.899, teff=5438,
           depth=0.0026, duration_hr=16.8, window=(1853.5, 1855.5)),
]

# BLS output on the candidates, 0.5-27 d search grid (from the Phase 3 run).
BLS_CANDIDATE_PERIODS = {
    233577004: 21.08,
    341687821: 19.38,
    122522333: 16.78,
    438122862: 14.36,
    232616346: 14.80,
}

def load_expanded(path=None):
    """Confirmed single-transit planets selected by scripts/12_select_targets.py.

    Returns [] if the selection has not been run yet.
    """
    import csv
    import pathlib

    path = pathlib.Path(path or pathlib.Path(__file__).resolve().parents[2]
                        / "results" / "data" / "expanded_targets.csv")
    if not path.exists():
        return []

    out = []
    for r in csv.DictReader(path.open()):
        def f(key):
            v = r.get(key, "")
            try:
                return float(v) if v not in ("", "None", "nan") else None
            except ValueError:
                return None

        depth = None
        if f("ratror"):
            depth = f("ratror") ** 2
        out.append(Target(
            tic=int(r["tic"]), sector=int(r["sector"]) if r.get("sector") else None,
            mass=f("st_mass"), radius=f("st_rad"), teff=f("st_teff"),
            depth=depth, duration_hr=f("pl_trandur"), true_period=f("pl_orbper"),
            name=r["pl_name"], known_b=f("pl_imppar"), known_ecc=f("pl_orbeccen"),
            known_k=f("ratror"), duration_kind="14", tranmid=f("pl_tranmid"),
            period_err=f("pl_orbpererr1"), tranmid_err=f("pl_tranmiderr1"),
        ))
    return out


EXPANDED = load_expanded()

ALL = {t.tic: t for t in VALIDATION_ONLY + CONFIRMED + CANDIDATES}
ALL_WITH_EXPANDED = {**ALL, **{t.tic: t for t in EXPANDED}}
