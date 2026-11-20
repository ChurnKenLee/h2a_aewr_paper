# Identification search

Scripts for `markdowns/identification_search_agent_plan.md`. They are
exploratory, not pipeline stages: they read retained intermediates and write
only to a scratch directory outside the repository. Nothing here changes a
producer, a retained output or the manuscript.

## Layout

| Path | Owner | Contents |
| --- | --- | --- |
| `shared/declarations.json` | Phase 0 lead | Frozen declarations: D1–D6, estimand, outcomes, cost unit, gate, pooling and multiplicity rules, `ledger_row.json` fields. SHA-256 in `shared/declarations.sha256` |
| `shared/01_outcomes.R` | Phase 0 | County × FY, employer × FY and county × calendar-year outcome panels, plus the case-row table |
| `shared/02_power_harness_test.R` | Phase 0 | Reproduction test of the harness against `explorations/identification/04_track1_power.R` |
| `shared/03_cost_unit.R` | Phase 0 | H-2A cost per hour: national FY ratio series and a state × FY series |
| `shared/power_harness.R` | Phase 0 | `ish_power()` and helpers (source it; no I/O) |
| `shared/split_helper.R` | Phase 0 | `ish_split()`: seeded one-third / two-thirds unit split |
| `shared/sources/cost_unit_sources.csv` | Phase 0 | Sourced non-wage cost components (citation, URL, retrieval date, confidence) |
| `ledger.csv` | lead and synthesis | One row per strategy, assembled by `synthesis/02_ledger.R` from the `ledger_row.json` files, the reviews and the triage files; columns are the declared `ledger_row_fields`. D1–D6 are recorded in `shared/declarations.json` (a CSV cannot carry a preamble) |
| `synthesis/01_pooling_multiplicity.R`, `synthesis/02_ledger.R` | Phase 7 synthesis | Hash checks, the declared pooling and multiplicity rules, the Track 1 reference conversion, and `ledger.csv`. Outputs in `$S/synthesis`; report in `markdowns/identification_search_report.md` |
| `<strategy_id>/` | each strategy | Registry, scripts, `ledger_row.json`, `estimates.csv`, `review.md` |
| `sources/` | Phase 2 acquisition | Sourced tables and notes for acquired data. A2: `aewr_2009_oes_schedule.csv`, `oes_area_county_crosswalk_2009.csv`, `h2a_disclosures_fy2006_2007.csv`, `aewr_2009_oes_verification.csv`, `aewr_2009_oes_notes.md` (built by `A2/00`–`02`; raw files in `data/raw/aewr_2009_oes/` and `data/raw/h2a_disclosures_fy2006_2007/`). A3: `aewr_2023_2025_occupation_rates.csv`, `aewr_2023_2025_oews_vintages.csv`, `aewr_2023_2025_oews_field_group_rates.csv`, `aewr_2023_2025_regime_periods.csv`, `aewr_2023_2025_verification.csv`, `aewr_2023_2025_offer_match_by_soc.csv`, `aewr_2023_2025_occupation_notes.md` (built by `A3/00`–`01`; raw files in `data/raw/aewr_2023_oews_occupation/`) |

## Run order

Run from the repository root. `$S` is the scratch root
(`/tmp/claude-1000/-home-ken-Dropbox-projects-H-2A-Paper/1be305e6-0442-48c5-93f4-7023961621fb/scratchpad/identification_search`
for this run); shared outputs go to `$S/shared`, strategy outputs to `$S/<strategy_id>`.

```sh
export R_LIBS_USER=$(ls -d "$PWD"/renv/library/*/R-*/* | head -1)
mkdir -p "$S/shared"
Rscript --vanilla explorations/identification_search/shared/01_outcomes.R "$S/shared"        # ~8 min
Rscript --vanilla explorations/identification_search/shared/03_cost_unit.R "$S/shared"       # needs 01
Rscript --vanilla explorations/identification_search/shared/split_helper.R "$S/shared"       # standard splits
Rscript --vanilla explorations/identification_search/shared/02_power_harness_test.R "$S/shared" \
  /tmp/claude-1000/-home-ken-Dropbox-projects-H-2A-Paper/1be305e6-0442-48c5-93f4-7023961621fb/scratchpad/identification   # needs 01
git checkout -- Rplots.pdf
sha256sum explorations/identification_search/shared/declarations.json   # must match declarations.sha256
# Phase 7 synthesis (after every strategy has finished); $T1 is the Track 1 scratch root (.../scratchpad/identification)
Rscript --vanilla explorations/identification_search/synthesis/01_pooling_multiplicity.R "$S" "$T1"
Rscript --vanilla explorations/identification_search/synthesis/02_ledger.R "$S"
```

## Shared outputs (`$S/shared`)

- `outcomes_county_fy.parquet`: `county_fips` × `year` (federal FY 2008–2026),
  balanced over every county in `county_year_panel` plus any H-2A county.
  Columns are `<measure>_<window>`. Measures: `cert_pos`, `cert_hours`,
  `cert_posweeks`, `req_pos`, `req_hours`, `apps`, `hourly_hours`,
  `hourly_aewr_known_hours`, `at_aewr_hours`, `share_at_aewr`, `n_employers`,
  `n_emp_entrants`, `n_emp_exiters`, `any_cert`, `entry_cert`, `exit_cert`,
  `any_req`. Windows:
  - `fy`: disclosure FY, all decisions (b01 logic; NA in FY2026);
  - `dq13`: disclosure FY, decisions 1 Oct–30 Jun;
  - `rfy`: FY of receipt, all decisions (NA in FY2026);
  - `rq13`: received 1 Oct–30 Jun, any decision date (NA in FY2026);
  - `rq13d`: received and decided 1 Oct–30 Jun (complete in FY2026; the
    lead window for requested positions).
- `outcomes_employer_fy.parquet`: `employer_id_balanced` × `year`, same
  measures and windows, plus `n_counties`, `n_states`, `first_fy_cert`,
  `last_fy_cert`.
- `outcomes_county_calendar.parquet`: county × calendar start year 2007–2026
  (b01 start-year logic); `year_complete` is TRUE for 2008–2025.
- `outcomes_case_rows.parquet`: the allocation rows behind every panel
  (case × county × employer, measures already multiplied by `alloc_weight`,
  with receipt, decision and period dates, SOC, hourly offer and the
  at-AEWR flag). Use it for job-order or monthly designs.
- `outcome_*.csv`: validation against `h2a_data_year`, requested coverage,
  `nbr_workers_needed` coverage, data quality by FY, national totals.
- `cost_unit_by_fy.csv`, `cost_unit_summary.csv`,
  `cost_per_hour_state_fy.parquet`: the cost unit.
- `split_*.csv`: standard state and county splits (seed 20261102).
- `harness_test_*.csv`, `harness_test_verdict.json`: harness test.

Requested positions are NA where the disclosure year lacks them (decision
FY2008–10 and FY2013–14) and in receipt windows contaminated by those years
(`outcome_requested_coverage.csv`). `nbr_workers_needed` exists only from
FY2020 and cannot fill those years.

## Power harness

```r
source("explorations/identification_search/shared/power_harness.R")
res <- ish_power(policy, pre, y = "y", x = "dose", unit = "county_fips", period = "year",
  fe = c("county_fips", "region_year"), clusters = c("state_fips", "aewr_region_id"),
  jk_cluster = "state_fips", node = "state_fips", group = "aewr_region_id",
  weights = NULL, beta_e1 = mean_y / mean_price, loading = ish_spatial_loading(states, edges, 0.4),
  ri = list(assign = "state_fips", strata = "aewr_region_id", P = 199), label = list(strategy = "A1"))
res$support   # dose SD after FE, clusters with dose, effective clusters, max leverage share
res$mde       # MDE (elasticity) by DGP x method x sign, size gate
res$summary   # size-valid MDE per DGP: first gate-passing method in the order CRV1, jackknife, RI
```

- `policy` is the balanced unit × period design frame with the dose; `pre`
  holds the calibration outcome with the same unit, period, FE, cluster and
  node columns. The scale comes from FE-only residuals of `pre[[y]]`.
- Build interacted FE as columns first (for example `paste(region, year)`).
- Settings come from `code/designs/border_discontinuity/design.json` (seed
  201103, 999 repetitions, AR(1) 0.5, upper-Wilson null rejection ≤ 0.075,
  power 0.80). The jackknife is the exact leave-one-cluster-out refit.
- RI permutes assignment-unit dose paths within strata (seed 20261102) and
  runs only when the dose is constant within assignment unit × period.
- `ish_node_period_share()` gives the share of pre-period residual variance
  at node × period cells next to the pure-unit-noise expectation; the
  declarations use it to pick the gate DGP.
- **PPML and hours.** For size-weighted PPML on counts (positions or hours),
  call `ish_ppml_linearize(pre, policy, y_count, unit)` and pass its `pre`,
  `policy`, `y` and `weights` with `beta_e1 = 1 / mean_price` (weighted mean
  price). Near the null the PPML score with unit FE equals WLS of
  `y / mu_i - 1` on the dose with weights `mu_i` (the unit's calibration-year
  mean count), so the simulated coefficient is the PPML semi-elasticity. For
  an aggregate elasticity (region or state totals), aggregate first and use
  the aggregate unit as `unit`. For hours per fixed denominator (the
  equal-unit ratio), use the level outcome with `beta_e1 = mean_y / mean_price`.

## Split helper

```r
source("explorations/identification_search/shared/split_helper.R")
ish_split(units = st$state_fips, strata = st$aewr_region_id)   # data.frame(unit, stratum, part)
```

Seed 20261102; one third (rounded, at least one per stratum of 3+) to
`exploration`, the rest to `confirmation`. Warns below 30 units: the plan
then skips the split and flags the strategy.
