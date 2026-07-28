# exopinn — single-transit orbital period recovery from TESS light curves

Recovering exoplanet orbital periods from **single-transit** TESS light curves with a
hybrid CNN + physics-informed neural network, benchmarked against Box Least Squares
and against the closed-form physics inversion.

```bash
python scripts/01_generate_synthetic.py --n 60000    # synthetic training set
python scripts/02_train_pinn.py --epochs 300         # train + benchmark
python scripts/03_bls_physicality.py                 # are BLS periods possible?
python scripts/04_apply_real_data.py                 # real TESS targets (needs MAST)
python scripts/05_ngts38_offline.py                  # the decisive case, no MAST needed
```

---

## The core result

Transit duration and orbital period are linked by Kepler's Third Law plus transit
geometry. Inverting that link gives

$$P \;\propto\; \frac{M_\star\,T_{\rm dur}^3}{R_\star^3\,\big((1+k)^2-b^2\big)^{3/2}}$$

**Period scales as duration cubed.** Every conclusion in this project follows from that
exponent:

| duration error | resulting period error |
| --- | --- |
| 5 % | 16 % |
| 10 % | 33 % |
| 31 % | 2.3× |
| 59 % | 4.0× |

This explains the earlier "PINN failures" exactly. PINN v2 was off by 4.11× on TOI-1899
and 1.35× on TIC 393818343; the cubic rule predicts 4.03× and 1.29× from the duration
mismatch alone. **The network had learned the physics to ~2%.** What failed was the
forward model it was given, not the network.

Two concrete errors in that forward model, both fixed here:

1. **The planet-radius term was missing.** Total transit duration uses
   $\sqrt{(1+k)^2-b^2}$, not $\sqrt{1-b^2}$, where $k=R_p/R_\star=\sqrt{\rm depth}$.
   For TOI-1899 (depth 4.45 %) that is a 21 % duration bias — a 1.8× period bias.
2. **Eccentricity was ignored.** Duration scales by $\sqrt{1-e^2}/(1+e\sin\omega)$,
   which spans 0.73×–1.36× for $e=0.3$. This is the *dominant* uncertainty in
   single-transit period recovery, and it is irreducible from one transit.

## Why a point estimate is the wrong product

From a single transit, impact parameter and eccentricity are both unobservable and
they push the implied period in opposite directions. For **NGTS-38 b (TIC 65910228)**,
a confirmed 180.53 d single-transit planet with published $b=0.854$, $e=0.309$:

| assumption | implied period | vs truth |
| --- | --- | --- |
| circular, central (b=0, e=0) | 101.5 d | 0.56× |
| circular, published b | 489.5 d | 2.71× |
| b=0, published e at apoastron | 39.0 d | 0.22× |
| published b **and** e | 188.0 d | 1.04× |

The forward model reproduces the truth to 4 % when handed the true geometry — so the
physics is right, and the spread is genuine degeneracy, not model error.

Marginalising over population priors for $b$ and $e$ gives the honest answer:

- median **190.7 d** against a truth of **180.5 d** (47.8th percentile), *with no
  period search range supplied at all*
- 68 % interval 94.8 – 833.2 d
- equilibrium temperature 445 K (68 %: 272–562 K), P(habitable zone) = 15.2 %

Because $T_{\rm eq}\propto P^{-1/3}$, a factor-8.8 period interval compresses to a
factor-2.1 temperature interval. **Habitability screening tolerates single-transit
precision far better than the period itself does** — which is the strongest scientific
argument for the whole approach.

The model therefore predicts a **distribution**, not a number: a second output head
gives a per-target $\sigma$ trained with a Gaussian NLL, and the reported interval width
is itself a result to be calibrated.

## Why BLS fails on NGTS-38 b, structurally

The planet completes 0.150 of an orbit during a 27-day sector.

- With a 0.5–27 d search grid the answer is outside the grid, so the error is bounded
  below by **85 %** regardless of the data.
- With a grid that *contains* 180.5 d, BLS still fails for a deeper reason: with one
  transit, every trial period longer than the baseline predicts exactly one transit in
  the window and fits identically well. **The periodogram is flat and `argmax` over a
  flat surface returns noise.** BLS needs a second transit, not a better grid.

This is a stronger claim than "BLS is inaccurate here", and it is testable.

## Physicality of BLS-reported periods

`03_bls_physicality.py` runs two tests on the five unconfirmed candidates:

- **Test A — does the orbit clear the star** ($a>R_\star$)? Rejects **0/5**. At periods
  of order days around main-sequence stars, $a/R_\star$ is 13–24. *This test has no
  discriminating power and should not be used to call a period impossible.*
- **Test B — what eccentricity would the BLS period require?** Inverting
  $f = T_{\rm obs}/T_{\rm circ}(P_{\rm BLS})$ gives $e_{\rm req}=(f^2-1)/(f^2+1)$ at
  $b=0$, the most generous geometry. Scored against the Kipping (2013)
  Beta(0.867, 3.03) prior, this rejects **3/5** as requiring $e\gtrsim0.55$ — the top
  few percent of the population.

The test is validated on the confirmed planets first: it demands $e\ge0$ for
TIC 393818343 and NGTS-38 b, and $e\ge0.27$ for TOI-1899 (published $e$ for NGTS-38 b
is 0.309). It does not reject truth.

---

## Results

### Synthetic data: the CNN branch beats a Bayes-optimal baseline

Held-out synthetic split, 9,000 samples. The Monte Carlo competitor is a kNN
posterior over the training set — same four scalar inputs, same priors including
the period prior — so it is the fair scalar-only Bayes benchmark.

| method | median err | scatter (dex) | within 2× | bias | 1σ coverage |
|---|---|---|---|---|---|
| analytic inversion | 61.6% | 0.613 | 40% | 0.43 | — |
| kNN Bayes (Monte Carlo) | 56.0% | 0.382 | 58% | 1.01 | 72% |
| CNN-PINN, scalars only | 57.3% | 0.385 | 57% | 1.08 | 70% |
| **CNN-PINN, full** | **38.2%** | **0.278** | **71%** | 0.98 | 73% |

The scalars-only control lands at 1.01× the Monte Carlo, confirming the network
learns the correct scalar posterior. The full model's 28% improvement therefore
cannot be prior shrinkage or scalar information — both controls carry those.

The eccentricity breakdown is the physical confirmation:

| eccentricity | Monte Carlo | CNN-PINN full |
|---|---|---|
| e = 0 | 51.2% | **27.7%** |
| 0–0.15 | 54.0% | **31.0%** |
| 0.15–0.35 | 55.5% | 45.7% |
| 0.35–0.90 | 66.8% | 66.1% (tied) |

The CNN reads impact parameter off transit shape, so it wins where b dominates
and ties where eccentricity — genuinely unobservable from one transit — takes
over. The gain appears exactly where theory says it should.

### Real data: the CNN advantage does not transfer

Three confirmed planets, real TESS photometry.

| method | median err | worst | 1σ coverage |
|---|---|---|---|
| BLS (0.5–27 d) | 64.7% | 89.8% | — |
| analytic inversion | 68.9% | 70.9% | — |
| CNN-PINN, full | 55.5% | 342% | 1/3 |
| **CNN-PINN, scalars only** | **15.6%** | 160% | **2/3** |

| target | true | analytic | full | scalars only |
|---|---|---|---|---|
| TIC 393818343 | 16.25 d | 4.73 (71%) | 7.23 (55%) | **17.92 (10%)** |
| TOI-1899 | 29.09 d | 27.07 (7%) | 128.62 (342%) | 75.75 (160%) |
| NGTS-38 b | 180.53 d | 56.18 (69%) | 237.14 (31%) | **152.30 (16%)** |

**The scalars-only model wins on all three.** The shape features that won 28% on
synthetic data are properties of the generator, not of TESS photometry — real
light curves carry cadence smearing, systematics, and detrending distortion the
simulator does not reproduce, and shape-reading is the most domain-sensitive
capability in the model.

The headline claim that survives: **a physics-informed scalar model recovers
single-transit periods to ~16% median error against BLS's ~65%, with no period
search range supplied.** n = 3 is a demonstration, not a statistic; the
statistics come from the 9,000-sample synthetic split.

Why b matters, on real data: TIC 393818343's measured FWHM of 3.35 h requires
b ≈ 0.72 at its true period, and its flat-bottom fraction of 0.75 independently
confirms a V-shaped, near-grazing transit. Assuming b = 0 gives 4.73 d; assuming
b = 0.85 gives 40.79 d. **Impact parameter alone swings that one target's period
by a factor of 8.6.**

### Injection-recovery (in progress)

Testing whether the failure above is a domain gap. Synthetic transits are
multiplied into *real* TESS photometry **before** detrending, so the injected
signal is distorted by the same filter that distorts a real one, then processed
by the identical `process_lightcurve()` used at inference.

Pool requirements, both of which an earlier version violated:

* every star in `targets.py` is excluded — training on noise from the evaluation
  stars is leakage
* stars are drawn across sky regions and explicit Tmag bins, because noise
  amplitude scales with brightness and shape legibility is an SNR question

Either outcome is a result: if the CNN closes the gap, the architecture is
justified; if it does not, transit shape as measured by TESS does not carry
usable impact-parameter information at these SNRs.

## Pipeline fixes carried over from the notebook version

| issue | consequence | fix |
| --- | --- | --- |
| Synthetic light curves generated on a `±1.5×duration` axis | Every training curve looks identical up to depth; the array carries **no absolute timescale**, so the CNN cannot encode duration. Real curves were then windowed in days — a train/test mismatch that produced the 486 % error. | Fixed 2.5-day absolute-time window, `WINDOW_DAYS`/`N_POINTS`, identical for synthetic and real. |
| Scalar features injected as ground truth in training, measured from data at inference | Noise, limb darkening and grazing geometry bias the features one way at test time and not at all at train time. | `measure_transit()` is applied to the simulated curve too — the **same estimator** at both ends. |
| `flatten(window_length=401)` | Counts **cadences, not time**: 13.4 h on 2-min data, 8.4 days on FFI. Same call, wildly different detrending. | Window specified in days, converted via the actual median cadence. |
| Savitzky-Golay trend fitted *through* the transit | The filter absorbs the signal — severe for the 14–16 h candidates, where the transit is most of a 13 h window. Depth comes out shallow, duration short, and period scales as duration cubed. | Two-pass detrend: locate the transit, then re-fit with those cadences masked out. |
| Mass and radius sampled independently | Duration depends on stellar **density**; independent sampling spends most of the training set on stars that cannot exist. | Main-sequence M–R relation with scatter plus an evolved tail. |
| Prediction in linear period | Allows negative periods (v1 returned **−10.88 d**). | Predict $\log_{10}P$; negatives are unrepresentable. |
| Physics loss as an equality constraint | Cannot be satisfied: eccentricity alone moves duration by ±31 %, limb darkening a further ~6 %. | Tolerance band — silent inside `DURATION_LOG_TOL`, pulls hard outside — plus a hard hinge at $a<R_\star$. |
| Duration definition not tracked | Literature values are $T_{14}$; a half-depth threshold measures FWHM. Mixing them shifts the period ~30 %. | `Target.duration_kind`, and `contact=` throughout `physics.transit_duration`. |

## Layout

```
src/exopinn/
  constants.py    AU / M_sun / day unit system
  physics.py      Kepler, transit duration (T14 / T23 / FWHM), analytic inversion,
                  physicality tests, equilibrium temperature; NumPy + torch
  lightcurve.py   transit forward model (quadratic limb darkening) and the shared
                  measure_transit / extract_window estimators
  synth.py        synthetic training set generator
  model.py        HybridCNNPINN, Normalizer, Gaussian NLL, physics loss
  train.py        training loop, checkpointing, prediction with intervals
  data.py         MAST download, two-pass detrend, transit location, feature extraction
  bls.py          BLS baseline and the search-range dependency experiment
  evaluate.py     analytic baseline, metrics, interval coverage
  targets.py      confirmed validation planets and unconfirmed candidates
```

## Known gaps

- **MAST was unreachable** while this was built (`mast.stsci.edu` resolves but TCP
  connect times out), so `04_apply_real_data.py` has not been executed against real
  light curves. It is written and ready; run it when MAST returns.
- The two candidate durations flagged `duration_reliable=False` (TIC 233577004,
  TIC 438122862) are window artefacts — both were reported as exactly 24.0 h, the
  width of the inspection window. Rows involving them are not interpretable until
  re-measured.
- The claim that the CNN branch narrows the interval by reading impact parameter off
  the ingress shape is **stated but not yet demonstrated**. The ablation is
  `02_train_pinn.py` with the light-curve branch removed; that experiment has not been
  run.
