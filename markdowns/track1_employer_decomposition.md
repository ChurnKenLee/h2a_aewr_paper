# 2025-rule design: is the offered-wage pass-through within employers?

- **Date:** 2026-11-13.
- **Script and registry:** `explorations/identification/10_track1_employer_decomposition.R`,
  `track1_employer_decomposition_registry.json` and its `.sha256` (same
  folder).
- **Outputs:** scratch directory
  `/tmp/claude-1000/-home-ken-Dropbox-projects-H-2A-Paper/0dae8120-5cae-40b7-a1f7-ef23357b9b56/scratchpad/t1emp`
  (called `$D` below). Every number here comes from a `t1emp_*.csv` written
  there; files ending `tau2026` hold the estimates and `tau2025` the placebo
  year.
- **Data read:** H-2A worksite rows for FY2023–FY2026, decisions dated
  1 October–30 June. No retained artifact, table or specification changed.

"The 2025-rule design" is the within-former-region contrast of
`markdowns/alternative_identification_exploration.md` §3 (called Track 1
there). Its first stage is that county mean offered wages fell $0.66 per
statutory dollar of cut.

## Verdict

- **The pass-through is within employers.** Of the 0.66, 0.62 is the same
  employer in the same county offering less. That is 93% of the total, with
  a 95% interval of 86% to 100%.
- **Composition contributes 0.05, and no part of it is distinguishable from
  zero.** Entry, exit and reweighting among continuing employers each move
  the slope by 0.03 or less.
- **Who files did not change with the size of the cut.** Exit, entry and the
  pre-change wages of employers that kept filing show no relation to the
  dose.
- **This says nothing new about H-2A use.** The variation is the 2025-rule
  design's own, and so are its limits: one season and about 11 effective
  states.
- **The trimming bounds are too wide to add anything.** They run from 0.29 to
  1.15 per statutory dollar.

## Order of work

1. The estimator was written and run for the placebo pair FY2024–FY2025 with
   the FY2026 dose. That run reads no row after FY2025.
2. The registry was written with the placebo results in it, and frozen with
   the estimator, its helpers and the base registry (five SHA-256 hashes).
3. The estimator was run once for FY2025–FY2026. It refuses to run unless all
   five hashes match.

Before freezing, the county-level FY2026 results of the 2025-rule design had
been seen, and so had the A1 review's employer-level wage changes by
at-AEWR share. No FY2026 row had been split into continuing, entering and
exiting employers.

You approved the analysis in outline. I chose the unit, component, bound and
test definitions, and you have not reviewed them individually.

## 1. Exact decomposition of the county first stage

**Cells.** Employer × county, using `employer_id_balanced`. A cell continues
if it has hourly offers in both years, exits if only in FY2025, and enters
if only in FY2026.

**Identity.** For each county, the change in the hours-weighted mean offered
wage equals the sum of five parts:

- **within:** the wage change of continuing cells at their FY2025 hours;
- **between:** reweighting among continuing cells;
- **entry:** entrants' FY2026 hours share × (entrant mean − continuer mean);
- **exit:** exiters' FY2025 hours share × (continuer mean − exiter mean);
- **no continuer:** the whole change in a county with no continuing cell.

Each part is regressed on the dose with former-region effects and equal
county weights, as in the 2025-rule design. The slopes sum to the total. The
code checks the identity in every county and checks the total against the
stored first-stage estimate; both hold (0.66233208).

| Part | Slope per statutory dollar | Jackknife SE | County mean, $ |
| --- | --- | --- | --- |
| Total | 0.662 | 0.095 | −2.18 |
| Within | 0.616 | 0.094 | −2.14 |
| Between | −0.008 | 0.016 | 0.01 |
| Entry | 0.021 | 0.033 | 0.01 |
| Exit | 0.027 | 0.033 | −0.02 |
| No continuer | 0.007 | 0.013 | −0.03 |

- 2,686 counties in 46 states.
- **Within share:** 0.930, jackknife SE 0.037 on the ratio.
- **Size of the continuing group:** 43,343 continuing cells hold 86% of
  FY2025 hours and 85% of FY2026 hours. 28,110 cells enter and 21,029 exit.
  34 counties have no continuing cell.
- **High-recall identifiers:** within 0.608 (0.096), share 0.918 (0.040).
  The other parts change by 0.005 or less.
- **Placebo year (FY2024–FY2025):** within 0.019 (0.027), between −0.004
  (0.006), entry 0.010 (0.050), exit −0.028 (0.031), total 0.003 (0.057).

## 2. Pass-through for continuing employers

Units are employer × state, since the dose is a state rate. The outcome is
the change in the unit's hours-weighted mean offer, on the dose with
former-region effects. 25,424 units have hourly offers in both years.

| Weights | Slope | Jackknife SE | Placebo year |
| --- | --- | --- | --- |
| Equal | 0.595 | 0.167 | −0.056 (0.079) |
| FY2025 hours | 0.644 | 0.091 | 0.012 (0.033) |

High-recall identifiers give 0.629 (0.156) and 0.647 (0.097).

## 3. Does who files vary with the cut?

**Exit and entry.** Linear probability models on the dose with former-region
effects. A positive coefficient means more exit where the cut was smaller.

| Margin | Base rate | Slope per dollar | Jackknife SE | Placebo year |
| --- | --- | --- | --- | --- |
| Exit, equal weights | 0.365 | 0.015 | 0.032 | 0.009 (0.024) |
| Exit, hours weights | 0.115 | 0.014 | 0.016 | 0.011 (0.011) |
| Entrant share, equal weights | 0.418 | 0.038 | 0.054 | −0.001 (0.020) |
| Entrant share, hours weights | 0.139 | 0.002 | 0.018 | 0.008 (0.008) |

- 40,008 FY2025 units and 43,647 FY2026 units.
- Defining exit as no certified positions of any wage unit changes the
  slopes by 0.002 or less.
- **Turnover is high but ordinary.** 36% of FY2025 units have no FY2026
  hourly offer, holding 11% of hours. The placebo year shows 36% and 15%.
  Some of this is failed linkage and some is the 30 June window.

**Did continuing employers already differ?** Among FY2025 units, two
pre-change outcomes are regressed on the dose, a stay indicator and their
interaction.

| Pre-change outcome | Stay | Stay × dose |
| --- | --- | --- |
| FY2025 offer minus FY2025 floor | 0.097 (0.320) | −0.004 (0.129) |
| FY2024-to-FY2025 wage change | −0.017 (0.061) | −0.005 (0.055) |

The interaction is the term that would bias the continuer estimate. It is
zero in both.

The textbook version of this test adds next year's filing indicator to the
fixed-effects regression. That is not available: the treatment varies only
in FY2026, and FY2027 filing is not yet observed.

## 4. Trimming bounds

Lee (2009) bounds on the continuer pass-through, within former region, with
states split into deeper and shallower cuts than their region's mean.

| Quantity | Estimate | Jackknife SE |
| --- | --- | --- |
| Binary-contrast point, per dollar | 0.72 | 0.26 |
| Lower bound | 0.29 | 0.31 |
| Upper bound | 1.15 | 0.49 |
| Placebo-year bounds | −0.15 to 0.13 | |

- The bounds trim 17% of the arm with more continuers in the average region.
  The average gap in continuation between arms is −1 percentage point; the
  17% comes from region-to-region differences in both directions.
- The placebo year trims the same share, so these differences are not caused
  by the cut.
- Each arm pools states with different cuts. Trimming a tail of wage changes
  then removes the states with the largest or smallest cuts, which moves the
  mean mechanically.
- The lower bound does not exclude zero. This is a limit of making a
  continuous state-level dose binary, not evidence of selection.

The evidence on selection is §3, not these bounds.

## What this changes

- The 2025-rule first stage can be described as employers lowering their own
  offers, not as a change in who files.
- A selection correction for the offered wage is not needed in this episode:
  filing did not respond differentially to the cut, and there is still no
  excluded variable that would identify one.
- The exit slope in §3 is also an extensive-margin usage estimate. Like the
  2025-rule design's positions estimate, it cannot be told from zero.

## Limits

- Every threat to the 2025-rule design carries over.
- "Within" holds the employer and county fixed, not the job. A continuing
  employer can change its mix of jobs.
- A failed employer link looks like an exit plus an entry. The high-recall
  run addresses missed merges only. The conservative tier cannot be mapped
  from the worksite rows and was not run.
- FY2026 cases received before 2025-10-02 carry the old rate and are not
  treated separately.
- Rows without an employer identifier (0.9% of FY2025 hourly hours, 0.2% of
  FY2026) count as exits and entries in §1 and are left out of §2–§4.

## Files

| File in `$D` | Contents |
| --- | --- |
| `t1emp_estimates_tau2026.csv`, `..._tau2025.csv` | Every regression: slope, jackknife SE and p, state CRV1 SE, sample sizes |
| `t1emp_within_share_*.csv` | Within share and its jackknife SE |
| `t1emp_county_means_*.csv` | County means of each part, continuing hours shares, cell counts |
| `t1emp_county_components_*.csv` | The five parts for every county (balanced identifiers) |
| `t1emp_selection_rates_*.csv` | Exit and entry rates by deeper and shallower cut |
| `t1emp_trimming_bounds_*.csv` | Bounds and their jackknife SEs |
| `t1emp_link_share_*.csv`, `t1emp_state_dose_*.csv` | Unlinked hours share by year; state doses |
| `tau2025.log`, `tau2026.log` | Run logs |
