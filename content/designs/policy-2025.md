+++
title = "2025 H-2A policy shift"
description = "Separate design for offered wages, Track 1 county H-2A quantities, mechanism heterogeneity, and covered crop earnings on the complete FY2026 H-2A disclosure, with the partial earnings window."
[extra]
scopes = ["code/designs/policy_2025", "scripts/run_policy_2025.sh", "outputs/reports/policy_2025.md"]
+++

`code/designs/policy_2025` owns the current wage-offer, Track 1 county H-2A
quantity, five-panel mechanism-heterogeneity, and covered-earnings analyses of
the October 2, 2025 shift. The user requested this separate design on November
16, 2026, as a migration after inspecting outcomes. On November 21, 2026 the
user made the complete FY2026 H-2A disclosure (Q4 release) the baseline and
released the design from preserving its earlier specification: the wage window
is now the full fiscal year and the H-2A exposure is rebuilt from that vintage.
On November 25, 2026, a dated minimum-wage vintage update
(`history/vintage_update_2026_11_25.json`) aligned this branch with the unified
A01 FRED minimum-wage series (`state_year_min_wage.parquet`), refreshed
`state_real_minwages.parquet` (restoring 11 federal-floor state-year rows:
GA/WY 2024--2026 and AL/LA/MS/SC/TN 2026) and `county_year_panel.parquet`,
added an explicit `!anyNA(r$ag_min)` completeness guard in `01_build_wages.R`,
and re-baselined only `tests/fixtures/wage_estimates.csv` on the 48
`schedule_envelope_*` reference-diagnostic rows. On November 26, 2026, a dated
additive amendment (`history/quantity_amendment_2026_11_26.json`) added stage
`11_estimate_quantity.R` (79 coefficient rows) to evaluate the complete FY2026
Q4 Track 1 county H-2A quantity response without altering the 22 pinned source
inputs or any of the 528 existing wage and earnings fixture rows
(`max_abs_diff <= 1.95e-14`); its headline `full_fy` PPML certified-hours
jackknife 95% lower bound anchors the separate [narrow welfare bounds
grid](@/designs/welfare.md). A second November 26, 2026 additive amendment
(`history/mechanisms_amendment_2026_11_26.json`) added stage
`11b_estimate_mechanisms.R` (96 mechanism-heterogeneity rows across five panels
with Holm step-down multiplicity control over 12 primary FY2025--FY2026
hypotheses) alongside code-generated publication LaTeX tables and figures,
pinning 2 additional cached inputs (`croplandcros_county_crop_type_acres.parquet`
and `h2a_with_fips.parquet`, 24 inputs total) while keeping all 607
pre-existing wage, quantity, and earnings rows invariant (`max_abs_diff <=
1.78e-15`). None of these updates is a new preregistration.
The [earlier chapter exploration](@/designs/chapter-directions.md) remains a
historical record; its 2013 administrative-interruption analysis is excluded.
The historical wage-reform and statewide-border power designs remain separate.

The registry and its historical declarations fix the estimands, source vintage,
weights and bounded queue. The branch README owns commands, stage order and
artifact locations. Its standalone runner consumes local source-owned inputs,
uses isolated temporary work, checks equivalence before publication and records
input/code/output hashes. It is outside the complete pipeline and manuscript
runners, while `code/welfare/` consumes its published wage and quantity CSVs by
pinned SHA-256. Neither the migration nor subsequent dated amendments promote a
causal claim.

{{ <grounding path="scripts/run_policy_2025.sh" anchor="policy-2025-runner" sha256="d42641a3a0db58aa50419820420202a9b722f90c19122feceee702e960e7eee0" /> }}

## Offered-wage estimand

The primary outcome is the nominal hourly offer change between FY2025 and
FY2026 for continuing R7 filer–state–occupation cells. Fiscal decisions use
October 1 through September 30 in every year (October through June before
November 21, 2026). FY2026 cases received on or after August 3, 2026 fall under
the second statewide schedule; the dose and reference rates stay those of
October 2, 2025. The dose is the U.S.-worker skill-I
statewide reference minus the FY2025 regional rate; a cut is negative.
Former AEWR-region effects and baseline certified-hours weights are retained.
Counties have positive fixed-2011 farm employment and belong to multi-state
former regions; Alaska, Hawaii and DC are excluded. Source hours retain the
historical pooled imputation and allocation convention.

Equal weights, R5/R8/disclosed FEIN, fixed baseline occupation composition,
within-filer across-state comparisons, 2024/2025 placebos, selection,
distribution and applicability audits remain in the complete ledger. Three
baseline heterogeneity interactions retain Holm-adjusted p-values and
Bonferroni simultaneous intervals. No specification is chosen by significance.

R7 identifies filers, not the source-owned worksite businesses. Shared name
normalizers are reused, while the design owns these unchanged filer rules.
Continuation and common occupations condition on observed survival; neither
defines a population effect without further assumptions. Exact job-level
legal floors stay unknown because receipt is not job-order submission and
skill, worker category, prevailing wage and itineraries remain unresolved.
The four-schedule/minimum-wage envelope is explicitly a reference diagnostic.
Following the November 25, 2026 minimum-wage unification, `01_build_wages.R`
asserts `!anyNA(r$ag_min)` after joining `state_real_minwages.parquet`,
restoring full state coverage in the 48 `schedule_envelope_*` diagnostic rows
(`44 -> 46` states in 2024--2025 and `39 -> 46` states in 2026; FEIN `tau = 2024`
remains `insufficient_support`) while leaving all 299 non-envelope wage rows,
all 135 pre-policy earnings rows, and all 46 interim earnings rows unchanged
(`max_abs_diff = 0.0` against the prior frozen fixtures).

## County H-2A quantity estimand (Track 1, FY2025--FY2026)

Stage `11_estimate_quantity.R` extends the frozen 2026-11-02 Track 1 county
quantity design to the complete FY2026 Q4 disclosure (full fiscal year
October 1 through September 30) across all `2,921` positive-2011-farm-employment
counties in the `15` multi-state former AEWR regions (`46` states; California
and Florida single-state regions are absorbed by `Former-Region x FY` effects).
The primary specification (`B_primary`) fits county-level `fixest::fepois`
(PPML) on certified hours and certified positions—alongside OLS per-2011-worker
ratios and direct `ln(rate)` PPML—against the October 2, 2025 U.S.-worker
skill-I statutory AEWR (`x_us_skill_1`) with `County` and `Former-Region x FY`
fixed effects and exact delete-one-state jackknife (`CV3J`, $t(45)$) inference.
Because the national level shift is absorbed, only within-former-region
dispersion in the statewide rate identifies the slope.

On the complete FY2026 Q4 disclosure (`full_fy`), the primary size-weighted
PPML certified-hours statutory elasticity is **`-0.078`** (`-0.0778`, CV3J SE
`0.2585`, 95% jackknife CI **`[-0.598, +0.443]`**, MDE `0.72`, `G = 46`).
Completing Q4 tightens precision by 40% relative to the matched `Q1--Q3`
benchmark (`-0.198`, 95% CI `[-1.004, +0.609]`, MDE `1.12`) and rules out
unit-elastic (`-1.0`) certified-hours responses at 95% confidence. Secondary
full-FY endpoints align with that bound: direct `ln(rate)` PPML hours `-0.062
[-0.452, +0.328]` (MDE `0.54`), PPML certified positions `+0.064 [-0.312,
+0.440]` (MDE `0.52`), OLS hours per 2011 worker `+0.261 [-0.722, +1.244]`, OLS
positions per 2011 worker `+0.544 [-0.721, +1.808]`, and a strong county
offered-wage first stage of `+0.6714` (CV3J SE `0.0988`, 95% CI `[0.47, 0.87]`).

Legal-cutoff sensitivity windows (`pre_aug03_decision` `-0.167 [-0.821,
+0.486]`, `pre_aug03_receipt` `-0.180 [-0.826, +0.466]`, `pre_aug26_decision`
for the August 26 *UFW* vacatur `-0.178 [-0.789, +0.433]`, and
`pre_aug26_receipt` `-0.113 [-0.546, +0.320]`) and pre-policy `FY2024--FY2025`
placebos (`full_fy` PPML hours `-0.033 [-0.293, +0.228]`, positions `-0.020
[-0.197, +0.157]`, first stage `+0.018`; `Q1--Q3` hours `-0.079 [-0.605,
+0.447]`) stay tightly centered near zero. All `S1`--`S5` sensitivities retain
full 46-state support via the unified `state_real_minwages.parquet` series
(`S3` H-2A skill-I `-0.081 [-0.521, +0.359]`, U.S. skill-II `+0.181 [-0.597,
+0.959]`; `S4` pre-Oct 2 cohort `-0.063 [-0.604, +0.479]`, dual-schedule
Oct 2/Aug 3 cohort `-0.044 [-0.583, +0.494]`; `S5` controlling for general
minimum wage `-0.061 [-0.573, +0.450]` or applicable agricultural minimum wage
`-0.084 [-0.603, +0.436]`; `S1` static FY2012--FY2026 `+0.126 [-1.672,
+1.924]`). Post-hoc total-cost-elasticity conversions (`cost_ratio = 1.2401`)
yield `-0.162 [-1.246, +0.922]` (MDE `1.51`) under the R7 offered-wage
pass-through (`0.6229`) and `-0.150 [-1.149, +0.848]` under the paired-jackknife
county first stage (`0.6714`). Because Track 1 uses one post-policy fiscal year
after outcomes had already been inspected, its headline hours estimate serves
as a credible one-season lower bound on elastic demand rather than a tightly
pinned structural point estimate.

## Mechanism heterogeneity (FY2025--FY2026)

Stage `11b_estimate_mechanisms.R` (`history/mechanisms_amendment_2026_11_26.json`)
decomposes the Track 1 county response across five pre-declared mechanism
panels (`96` rows: `48` `FY2025--FY2026` estimates paired with `48` matched
`FY2024--FY2025` placebos) on the identical `2,921`-county, `46`-state,
`15`-former-region frame (`County` and `Former-Region x FY` FE, exact state
deletion CV3J jackknife `t(45)`, paired 46-state jackknife contrasts, and
exact hours/positions conservation checks). Twelve primary `FY2025--FY2026`
hypotheses are jointly adjusted via Holm step-down $p$-values
(`min p_holm_family = 0.939`):

- **Panel A (Application pipeline)**: All-filed `requested_hours` `+0.058
  [-0.607, +0.724]` (`MDE 0.93`, Holm `p = 1.000`, placebo `-0.114 [0.190]`),
  `requested_positions` `+0.162 [-0.328, +0.652]`, `approval_rate` `-0.0062
  [-0.0228, +0.0104]` per dollar, and paired `requested - certified` hours
  `+0.136 [-0.322, +0.594]`. Short-run inelasticity is not an OFLC/SWA denial
  bottleneck.
- **Panel B (Incumbent capacity vs. employer entry, `employer_id_balanced`,
  1-year lookback)**: `incumbent_1yr_hours` (`86.2%` of FY2026 Track 1 hours)
  `-0.086 [-0.866, +0.694]` (placebo `+0.113 [0.230]`), `entrant_1yr_hours`
  `-0.213 [-4.275, +3.849]`, paired diff `-0.127 [2.370]`, distinct worksite
  businesses `+0.444 [-1.070, +1.958]`.
- **Panel C (H-2ALC contractors vs. fixed-site direct growers)**: Fixed-site
  direct growers (`74.7%` of Track 1 hours; `65.9%` nationally) deliver the
  tightest elasticity bounds in the program: **`direct_grower_hours = -0.066
  [-0.440, +0.308]`** (`MDE 0.52`, Holm `p = 1.000`, placebo `+0.064 [0.186]`)
  and **`direct_grower_positions = -0.021 [-0.293, +0.251]`** (`MDE 0.38`),
  whereas `h2alc_hours` (`-0.103 [-2.417, +2.210]`, placebo `-0.602 [0.485]`)
  are noisy and `h2alc_hours_share` is flat (`+0.0016 [0.0044]`).
- **Panel D (Planning lead-time & predetermined `2022--2024` CDL crop
  capital)**: Late spring/summer starts (`Apr--Sep`, `36.9%` of FY2026 Track 1
  hours) yield `-0.676* [-1.431, +0.080]` (`p_holm = 0.939`, placebo `-0.286
  [0.150]`) vs. early winter starts (`Oct--Mar`) `+0.055 [-0.992, +1.101]`
  (paired diff `-0.730 [0.820]`, `p_holm = 1.000`). Predetermined CDL perennial
  fruit/tree-nut shares show no differential response (continuous within-region
  interaction `+0.012 [-0.793, +0.817]`, `p_holm = 1.000`; annual-dominated
  `< p75` counties `+0.020 [-0.499, +0.540]`), while an explicitly secondary
  exploratory continuous vegetable-share interaction is `-0.230 [0.047]`
  (placebo `-0.071 [0.053]`, quartile contrast `-0.567 [0.531]`).
- **Panel E (Intensive contract margins)**: Implied `ln(hours/position)`
  `-0.141 [-0.503, +0.220]` (`MDE 0.50`), `mean_contract_weeks` `-0.043
  [-0.304, +0.218]`, and observed non-imputed `mean_weekly_hours` `+0.016
  [-0.044, +0.077]` (`MDE 0.08`).

## Covered-earnings estimand

The current primary is the user-authorized November 14 interim window:
the mean of the 2025Q4 and 2026Q1 year-over-year log changes in private crop
production average weekly earnings. Each county must have both current/prior
pairs, positive disclosed earnings and employment of at least ten in all four
observations. States require at least two eligible counties. Missing periods
stop the requested window; missing county observations exclude the county
from that window and are never imputed.

{{ <grounding path="code/designs/policy_2025/interim_helpers.R" anchor="policy-2025-interim-window" sha256="27e55643bb8c0dbfd9870c66c677fe945a396b263cda54467dfb249830cd921d" /> }}

Intensity is mean 2022–2024 H-2A certified positions per fixed-2011 farm
employment, winsorized at its 99th percentile. The county exposure/dose table
was rebuilt on November 21, 2026 from the FY2026 Q4 panel with the unchanged
`explorations/episode_2025/09_domestic_wage_exposure.R`; it differs from the
earlier table in 42 Northeastern counties and has SHA-256
`848295a07cd1a5d21a91d115c9a1ee958d1bab7935799da6e15764b10b865669`.
The regressor is intensity times the statutory dollar cut, with intensity and
state effects included and equal county weights. There is no added trend,
instrument, normalization, seed, new resource guard or sample restriction.

Employment weights, H-2A cash dose, commuting-zone exposure and omission of
the intensity control remain sensitivities. The design also retains separate
quarters, the same two-quarter sample in each quarter, the full prior-window
lead, matched Q4/Q1 placebos for 2016–2024, the earlier three-quarter pre-policy
diagnostics, frozen UI splits, animal and all-private comparisons. QWI is
secondary and unavailable unless the same declared periods are present.

The earlier full target is 2025Q4–2026Q2 with all six prior/current observations.
Its implementation and publication guard are preserved, and its historical
acquisition command remains separately documented. It is not invoked by the
current runner. The interim and any later full-window update are multiple
looks at already inspected outcomes, not independent confirmations.

{{ <grounding path="code/designs/policy_2025/helpers.R" anchor="policy-2025-complete-window" sha256="51503cc63e3c408ca15f7621b3026f35d2590fce8a7127d37bf993aca976e50c" /> }}

## Inference and interpretation

All migrated fits retain state deletion jackknife inference with t(G−1)
intervals and CRV1 as a secondary comparison. Any unidentified deletion leaves
the jackknife interval unavailable; failed replicates are not silently dropped.
Every failed fit remains in the ledger. The fixed-effects small-sample
conventions and one-thread setting are explicit in this branch's helpers.

Causal interpretation needs exposure-related untreated changes to be unrelated
to the dose after controls and fixed effects. Differential state shocks and
contemporary policy changes remain concerns. The initial wage extensions were
developed after related FY2026 findings were seen. Quiet pre-policy diagnostics
cannot establish exogeneity. Weekly earnings mix wage rates, hours and workforce
composition; incomplete UI coverage cannot establish domestic-worker effects.
No welfare elasticity or general labor-market spillover follows automatically.

## Validation and ownership

The runner requires all 24 recorded source hashes (`sources/input_vintage.json`),
pins exposure and UI classification, and compares all 703 coefficient rows (347
wage, 79 Track 1 quantity, 96 mechanism-heterogeneity, 135 pre-policy earnings,
and 46 interim earnings) to the reference ledgers recorded for the current
vintage at absolute tolerance 1e-10. Samples, formulas, weights, failed
statuses, p-values, Holm adjustments, standard errors and confidence intervals
are included. Independent base-R WLS/OLS, quantity PPML, and mechanism PPML
spot checks, `h2a_data_year.parquet` FY2012--FY2025 parity, exact subgroup
conservation checks, allocation conservation, CDF ordering, complete periods,
suppression and canonical string-key checks are retained (`356/356` validation
checks plus `12/12` interim checks).

The branch writes filer exchange artifacts to `data/intermediate/policy_2025`,
analysis panels (including `quantity_county_fy.parquet` and
`mechanisms_county_fy.parquet`) to `data/processed/policy_2025`,
estimates/audits and publication LaTeX tables
(`table_policy_2025_{wages,quantity,mechanisms,earnings}.tex`) to
`outputs/tables/`, publication figures (`fig_policy_2025_{wages,mechanisms}.{pdf,png}`)
to `outputs/figures/`, and reports/provenance to `outputs/reports/`. The
publisher verifies copied bytes across all 52 retained artifacts and records a
complete manifest. A further source refresh or empirical revision requires a
dated record under `history/`; it must not silently replace the comparison
ledgers or change an upstream schema in this consumer.
