# 2025–27 policy episode: pre-period checks and draft registries

- **Date:** 2026-11-12; updated 2026-11-13 after the user's decisions (see "Update, 2026-11-13").
- **Scripts and registries:** `explorations/episode_2025/` (see its README).
- **Outputs:** scratch directory
  `/tmp/claude-1000/-home-ken-Dropbox-projects-H-2A-Paper/7f9a2d7a-92fd-4191-9ce4-cc8038d6bc79/scratchpad/episode_2025`
  (called `$E` below). Every number here comes from a CSV written by the
  script named next to it.
- **Data read:** H-2A outcomes through FY2025 and pay outcomes through 2024.
  No post-policy outcome was read. No retained artifact, table or
  specification changed.
- **Data acquired, not opened:** QCEW quarterly files for 2025 and 2026, in
  `data/raw/qcew_episode_2025_2026/`.

"The 2025-rule design" is the frozen within-former-region contrast of
`markdowns/alternative_identification_exploration.md` §3 (called Track 1
there).

## Verdict

- **Neither idea for sharpening the usage design works.** Using the
  between-region part of the cut brings a dose-correlated pre-trend that
  adjustments do not remove. Range herders and piece-rate orders do not share
  the treated jobs' state-year shocks, so differencing against them adds
  noise.
- **The 2025-rule design should stay as it is, size-weighted.** Its placebo
  error is far smaller with size weights than with equal weights.
- **The August 2026 schedule is not a reversal.** Its within-region
  dispersion is about a third of the 2025 cut's. More precision depends on
  the replacement rule, whose rates are not published.
- **No later window adds usable precision.** Late-season FY2026 windows are
  far too thin, and a second season averaged with the first is no more
  precise than the first alone.
- **The domestic-wage design is feasible and blind.** It can detect pay
  changes of about 3% in a heavily exposed county. Its registry is frozen and
  its data are downloaded but unopened.
- **The usage registry stays a draft.** Nothing powered can be estimated
  under it before the first FY2027 disclosure.

## 1. Check A: the between-region part of the cut

Script `01_usage_full_cut_precheck.R`; outputs `$E/full_cut_*.csv`.

**Method.** For each placebo year 2018–2025, the state's one-year log growth
in certified hours or positions (decisions October–June) is regressed on the
actual 2025 dose. Nothing happened to the dose in those years, so the root
mean squared estimate is the design's error, trend bias included. Figures are
2.8 × that error in elasticity units. They compare designs; they are not
size-validated MDEs.

**Check.** The two-period state PPML reproduces the reported FY2024-vs-FY2025
placebo: −0.09 for hours and −0.13 for positions, against −0.08 and −0.11
reported (`$E/full_cut_ppml_validation_fy2024_fy2025.csv`).

**Support.** The full dose has a third more dispersion: hours-weighted SD
$1.10 across 48 states, against $0.82 within former regions across 46
(`$E/full_cut_support.csv`).

**Size-weighted results** (`$E/full_cut_placebo_summary.csv`):

| Design | Hours: bias | Hours: error | Positions: bias | Positions: error |
| --- | --- | --- | --- | --- |
| Within former region (current) | 0.04 | 0.41 | −0.03 | 0.45 |
| Full dose, no adjustment | −0.30 | 0.96 | −0.29 | 0.90 |
| Full dose, state trend removed | −0.29 | 0.91 | −0.13 | 0.53 |
| Full dose, two lags of growth | −0.29 | 0.90 | −0.31 | 0.97 |
| Full dose, two factor loadings | −0.29 | 0.91 | −0.29 | 0.89 |
| Full dose, trend removed + two loadings | −0.22 | 0.75 | −0.13 | 0.51 |

- **The full dose is correlated with state growth trends.** States with
  larger cuts were already growing faster, worth about −0.3 elasticity units
  a year. This is the pre-trend the earlier report found.
- **Adjustments remove about half of it at best.** The best full-dose design
  is still worse than the within-region design on both outcomes.
- **The factor structure is weak.** The first two factors hold 59% of past
  growth variance in-sample and do not reduce placebo error.

**Equal state weights.** The within-region design's error is 1.43 (hours) and
1.41 (positions); the best full-dose designs reach 0.90 and 0.70. That is an
improvement on a design that is itself much worse than the size-weighted one.

**Side finding.** The size-weighted within-region error (0.41–0.45) is well
below the realized jackknife MDE of the first season (1.12 for PPML hours).
Eight placebo years cannot settle this: the 95% range of the 0.41 figure is
roughly 0.28 to 0.79.

## 2. Check B: jobs the rule did not touch as within-state controls

Script `02_usage_untouched_jobs_precheck.R`; outputs `$E/untouched_*.csv`.

**Groups.** Range = monthly-paid job with a herder or range title (the
`herder_regex` of `code/designs/wage_reforms/design.json`). Piece = non-range
order with a positive piece-rate offer; that field exists from FY2020 only.

| | Range herders | Piece-rate orders |
| --- | --- | --- |
| FY2025 control hours (millions) | 3.5 | 85 |
| FY2025 treated hours (millions) | 330 | 246 |
| States in support | 9 | 35 |
| Share of treated hours in those states | 21% | 86% |
| Correlation of control and treated growth | −0.06 | −0.37 |
| Growth variance after differencing, relative to treated alone | 3.4× | 15× |
| Placebo error, double difference (size-weighted) | 1.78 | 0.54 |
| Placebo error, triple difference (size-weighted) | 1.91 | 1.48 |

- **Range herders are too few and do not co-move.** Three states hold 47% of
  range hours, and their growth is unrelated to non-range growth in the same
  state.
- **Piece-rate orders move against other orders.** The negative correlation
  suggests orders switch between listing and not listing a piece rate, so the
  two groups are not separate job populations.
- **Not tested:** whether piece-rate earnings moved with the cut. That needs
  FY2026 offers and is moot given the above.

## 3. Events and the usage registry

Script `03_event_support.R`; outputs `$E/event_dose_*.csv`. Draft registry
`explorations/episode_2025/usage_episode_registry_draft.json`.

| Event | Mean dose | Hours-weighted SD | Within former region |
| --- | --- | --- | --- |
| 2025-10-02 schedule (U.S. skill I) | −$3.96 | $1.07 | $0.81 |
| 2026-08-03 schedule (U.S. skill I) | +$0.48 | $0.45 | $0.28 |

Weights here are full-year FY2025 certified hours; §1 weights by
October–June hours, which is why its figures differ slightly.

- **The August 2026 change is small and unrelated to the cut.** Its
  correlation with the 2025 dose is −0.36 overall and 0.02 within former
  region. It adds about 12% to the first event's identifying variance.
- **The replacement rule is the event that matters.** Trade-press summaries
  say the court ordered one by 2026-12-30. No rates are published, so the
  registry leaves it to a dated amendment.
- **Windows are declared by receipt date,** so that cases filed after the
  2026-08-26 ruling, when back pay became possible, can be kept apart.

**Later windows** (script `07_usage_extension_precheck.R`; outputs
`$E/extension_*.csv`). Placebo-year error, size-weighted, within former
region, in elasticity units:

| Contrast | Share of a year's hours | Error |
| --- | --- | --- |
| October–June decisions, one year (current design) | 83–85% | 0.41 |
| Cases received 1 July – 2 August | 5–6% | 3.6 |
| Cases received 3 August – 30 September, August 2026 dose | 3–6% | 14.5 |
| Second season against the base year | | 0.59 |
| Mean of two seasons against the base year | | 0.47 |
| First season alone, same placebo years | | 0.43 |

- **The FY2026 Q4 disclosure adds nothing usable.** Its windows are declared
  secondary in the draft registry and are not pooled.
- **A second season answers the one-season caveat, not the precision
  problem.** Both seasons share the base year's noise.

**Legal facts** (`explorations/episode_2025/sources/legal_status_notes.md`).
Read from the order and the OFLC notice: the 2025 rule was held unlawful on
2026-08-26 and not vacated; DOL must promptly issue a new methodology; the
backpay period starts 2026-09-02; current rates stay in force. The reported
order of 2026-10-02 with a 2026-12-30 deadline was not verified.

**Decisions recorded** (user, 2026-11-13): size-weighted certified hours is
the primary outcome for unseen windows, and cases filed after the ruling are
not separated. **Still open:** whether to report a pooled two-season
estimate; the draft recommends reporting the seasons side by side.

## 4. Domestic-wage design

Scripts `04_domestic_wage_support.R` and `05_h2a_in_qcew_state.R`; outputs
`$E/domestic_wage_*.csv` and `$E/h2a_fte_vs_qcew_*.csv`. Draft registry
`explorations/episode_2025/domestic_wage_registry_draft.json`.

**Design.** Year-over-year change in log county crop pay on (H-2A intensity ×
state cut), controlling for intensity, with state effects. Intensity is
2022–24 certified positions per 2011 farm worker.

**Precision, placebo years 2016–2024, controlling for intensity**
(`$E/domestic_wage_placebo_summary.csv`):

| Series | Share of benchmark | Pay change at intensity 0.3 and a $4 cut |
| --- | --- | --- |
| QCEW crop pay per worker, annual | 0.65–0.90 | 3.1–4.3% |
| QWI stable-job earnings, annual | 0.46–0.76 | 2.2–3.6% |
| QWI stable-job earnings, Q4–Q2 window | 0.54–0.67 | 2.6–3.2% |
| QCEW average weekly wage, Q4–Q2 window | 0.61–0.80 | 2.9–3.8% |

The benchmark is crop pay moving with H-2A offers in proportion to intensity.
The ranges span employment and equal weights.

**Who is in the series.** In 2024, H-2A full-time equivalents exceed all
state crop and crop-support employment in Georgia, Kentucky, Louisiana and
North Carolina, and exceed crop employment alone in Arkansas, Mississippi and
South Carolina as well (28% of national H-2A full-time equivalents in all).
There is no state-level inconsistency in Florida, California or Washington
(`$E/h2a_fte_vs_qcew_excess_states.csv`). So the benchmark is generous where
H-2A workers are outside the records.

**Frozen 2026-11-13.** The registry is
`explorations/episode_2025/domestic_wage_registry.json`; hashes of it, the
exposure table, the estimator, the extractor and the helpers are in
`domestic_wage_registry.sha256`.

- **Primary series.** The user was unsure, so the draft's rule was applied to
  pre-period data: with equal weights the QCEW window (0.61 of the benchmark)
  is as precise as the QWI window (0.67), so QCEW is primary. The user can
  reverse this by amendment before any outcome is read.
- **Window.** 2025 Q4, 2026 Q1 and 2026 Q2, year over year. Nothing is
  estimated until the release containing 2026 Q2; the estimator stops if a
  quarter is missing.
- **Estimator check.** `08_domestic_wage_estimate.R` on placebo year 2024
  reproduces the support script exactly (−0.0081, state-clustered SE 0.0082).
- **Insurance-coverage split.** Declared, but its state table must be sourced
  from statute and hashed by amendment before estimation; otherwise it is not
  run. The primary pools all states.

**Acquired 2026-11-13** (user approval): `2025_qtrly_singlefile.zip` and
`2026_qtrly_singlefile.zip` from BLS, with hashes in
`data/raw/qcew_episode_2025_2026/MANIFEST.csv`. Both carry a BLS
last-modified date of 2026-08-21, so 2026 Q2 is not in them yet. They have
not been opened. **Not acquired:** QWI. Its producer needs the Census API
key and is fixed at 2024, and the window's quarters are unlikely to be
published yet; it is a secondary series.

## Limits

- Placebo-year errors rest on eight or nine years and are not size-validated.
- The order and the OFLC notice are archived under
  `explorations/episode_2025/sources/`. The order was taken from a
  third-party copy, and the later deadline order was not found.
- Certified hours overstate hours worked.
- Scripts 01–03 depend on the shared outcome panels of the identification
  search, which live in that search's scratch directory.

## Update, 2026-11-13

Sections 3 and 4 above carry the changes. In short: the user chose
size-weighted hours and no separation of post-ruling cases; the legal facts
were checked against the order; the extension pre-check showed that no later
window adds precision; the domestic-wage registry was frozen with QCEW as the
primary series; and the 2025 and 2026 QCEW quarterly files were downloaded
without being opened.
