# 2025 H-2A policy shift

Standalone design for wage offers and covered agricultural earnings around the
October 2, 2025 policy change. Read the [canonical design](../../../content/designs/policy-2025.md)
and [design registry](design.json) before changing behavior.

The branch began as the November 16, 2026 migration of the inspected wage and
earnings analyses from `explorations/chapter_directions`; that code, its
registries and reports remain historical records. Since the user's instruction
of November 21, 2026 the baseline is the complete FY2026 H-2A disclosure (Q4
release): wage offers use decisions over the full fiscal year (October 1
through September 30) instead of October through June, and the H-2A exposure
table is rebuilt from the same vintage with an unchanged definition. Earlier
specifications are not preserved; `history/vintage_update_2026_11_21.json`
records the change. On November 25, 2026 a unified minimum-wage vintage update
(`history/vintage_update_2026_11_25.json`) adopted the rebuilt
`state_real_minwages.parquet` (from unified A01 FRED producer
`code/a01_sources/07_state_minimum_wages.py`) and `county_year_panel.parquet`,
restoring `$7.25` federal-floor agricultural minimum wages for GA/WY
(2024–2026) and AL/LA/MS/SC/TN (2026), adding an explicit `!anyNA(r$ag_min)`
guard in `01_build_wages.R`, and re-baselining only the 48
`schedule_envelope_*` rows in `tests/fixtures/wage_estimates.csv` while keeping
all other wage-offer, pre-policy earnings, and interim earnings ledgers
unchanged. On November 26, 2026 (`history/quantity_amendment_2026_11_26.json`)
an additive Track 1 county quantity stage (`11_estimate_quantity.R`, 79 rows)
extended the frozen 2026-11-02 Track 1 quantity design to the complete FY2026 Q4
disclosure (`2,921` multi-state counties, `46` states, `15` former regions)
without altering the 22 pinned source inputs or any of the 528 existing wage
and earnings fixture rows. A second November 26, 2026 additive amendment
(`history/mechanisms_amendment_2026_11_26.json`) added stage
`11b_estimate_mechanisms.R` (`96` rows across 5 mechanism panels, with
Holm-adjusted `p`-values across 12 primary FY2025–FY2026 hypotheses) plus
code-generated manuscript LaTeX tables and figures, pinning 2 additional cached
local inputs (`croplandcros_county_crop_type_acres.parquet` and
`h2a_with_fips.parquet`, bringing `sources/input_vintage.json` to 24 inputs)
while keeping all 607 pre-existing wage, quantity, and earnings fixture rows
invariant. Outcomes had been inspected before it. It does not run the 2013 shutdown analysis or import another
design's numerical settings. Historical county decomposition/utilization and
state-border power exercises remain separately documented references.

## Run

From the repository root, in the existing devenv/renv environment:

```sh
DRY_RUN=1 ./scripts/run_policy_2025.sh
./scripts/run_policy_2025.sh
```

The standalone runner uses only cached public inputs. It is outside
`run_all.sh`, manuscript and welfare runners. It uses the shared pipeline
helper, exports Marimo applications into temporary `/tmp` storage, runs R
with `--vanilla`, validates results, publishes artifacts and cleans temporary
files. A dry run does not read data, import R/Python, or access the network.

On the initial migration, `POLICY2025_LEGACY_SCRATCH` may point to the earlier
chapter scratch directory to reuse exactly three hash-verified dependencies:
the two filer Parquets and the pre-policy QCEW extract. It does not import
estimates or administrative artifacts. Subsequent runs use the retained design
dependencies. If caches are absent, the design rebuilds them from local inputs.
`sources/input_vintage.json` checks all 24 source files; drift stops the run
and requires a documented vintage/specification update.

## Order and ownership

The original stage numbers are retained to make the migration inspectable.

| Stage | Responsibility |
| --- | --- |
| `00_prepare.py` | Verify registry, exposure/source vintage and reusable caches; rebuild filer inputs through `inputs/01_filer_records.py` and `inputs/02_rule_harness.py` when needed |
| `01_build_wages.R` | Build filer–state–occupation and certification panels, state dose, allocation and floor audits |
| `02_estimate_wages.R` | All 347 wage-offer fits, state deletions, composition, selection, heterogeneity and distribution checks |
| `05_extract_earnings.py` | Pre-policy three-industry QCEW extraction; preserve full-window calendar/publication gate |
| `06_estimate_earnings.R` | All 135 pre-policy fits, disclosure/quarter audit and frozen UI classification |
| `09_extract_interim.py` | Verify and extract cached 2025/2026 archives using the same corrected aggregation |
| `10_estimate_interim.R` | Interim, quarter, common-sample, lead, sensitivity and matched-placebo queue |
| `11_estimate_quantity.R` | Complete FY2026 Q4 Track 1 county quantity response (79 fits), Q1–Q3 benchmark, Aug 3/Aug 26 cutoffs, placebos, S1–S5 sensitivities, and `epsilon_cost` conversion |
| `11b_estimate_mechanisms.R` | Post-inspection FY2026 Q4 mechanism heterogeneity across 5 panels (96 fits, 12 Holm-adjusted primary hypotheses), `table_policy_2025_mechanisms.tex`, and `fig_policy_2025_mechanisms.{pdf,png}` |
| `tests/check_interim.R`, `07_validate.R` | Period/suppression/geography/conservation checks, independent OLS/WLS/PPML, full-ledger equivalence (703 rows) |
| `08_report.R` | Generated current report, complete specification appendix, `table_policy_2025_{wages,earnings}.tex`, and `fig_policy_2025_wages.{pdf,png}` |
| `12_publish.R` | Retain validated products, record input/code/output hashes and verify published bytes |

`inputs/` owns this design's unchanged R5/R7/R8 filer reconstruction. It uses
shared name normalizers without changing the canonical worksite-business
crosswalk. Frozen declarations under `history/` retain their original dates and
paths as historical evidence, not runtime imports or new preregistrations.
`registry.sha256` pins declarations, input vintage, exposure, UI review and
migration comparison ledgers. Runtime implementation hashes are recorded in
the generated provenance. Do not automatically refresh frozen hashes.

## Inputs and retained products

The branch consumes the shared county panel, allocated worksites, disclosure
records, membership records, AEWR schedules, agricultural minimum wages, QWI,
cached QCEW 2014–2024 quarterly ZIPs, and
`data/raw/qcew_episode_2025_2026/{2025,2026}_qtrly_singlefile.zip` with their
acquisition manifest. Filer reconstruction also uses the local raw disclosure
workbooks. No source acquisition or credentials are part of the default run.

| Location | Contents |
| --- | --- |
| `data/intermediate/policy_2025/filer/` | `filer_cases.parquet`, `filer_ids.parquet` |
| `data/processed/policy_2025/` | Wage occupation/certification panels; Track 1 county-FY quantity and mechanism panels; pre-policy and interim earnings panels |
| `outputs/tables/policy_2025/` | Wage, Track 1 quantity, mechanism, and earnings coefficient ledgers, state deletions, cutoff audits, cost conversions, mechanism descriptives, distributions, coverage, floor, allocation, quarter and validation audits |
| `outputs/tables/table_policy_2025_*.tex` | Booktabs summary tables for offered wages, Track 1 quantity, mechanism heterogeneity, and interim covered crop earnings |
| `outputs/figures/fig_policy_2025_*.{pdf,png}` | Publication figures for offered-wage pass-through (`fig_policy_2025_wages`) and mechanism heterogeneity (`fig_policy_2025_mechanisms`) |
| `outputs/reports/policy_2025.md` | Current integrated findings |
| `outputs/reports/policy_2025_specifications.md` | Every wage, pre-policy and interim coefficient row |
| `outputs/reports/policy_2025/` | Input/vintage metadata, validation JSON and `provenance.json` |

Apart from the wage window, regressions keep their nominal units, sample rules,
weights and inference. The interim primary remains **2025Q4–2026Q1**, requiring both
current/prior pairs; no missing quarter is imputed. The full three-quarter
target is not substituted for this window. QWI remains unavailable if its
declared periods are missing. The original guarded full-window workflow is
documented in [the historical exploration README](../../../explorations/chapter_directions/README.md).

## Validate

The runner validates all 703 coefficient rows (347 wage, 79 Track 1 quantity,
96 mechanism heterogeneity, 135 pre-policy earnings, and 46 interim earnings)
against the reference ledgers recorded for the current vintage at absolute
tolerance `1e-10`, including sample sizes, statuses, formulas, weights,
standard errors and intervals. It also runs the 12 interim checks,
`h2a_data_year.parquet` FY2012–FY2025 parity, mechanism conservation checks,
independent quantity and mechanism PPML, and independent wage WLS. Source-vintage changes must
preserve earlier results and declare new comparison expectations explicitly.

```sh
UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync ruff check code/designs/policy_2025
bash -n scripts/run_policy_2025.sh
python scripts/agent_docs.py verify
python scripts/test_agent_docs.py
zola check --skip-external-links
zola build
```

`tests/test_earnings.py` retains the extractor's seven edge checks. Export
`05_extract_earnings.py` under `/tmp`, set `POLICY2025_WORK` to external scratch
and `CD_EXPORTED_EARNINGS` to that exported file, then run the test through
`uv run --no-sync python`. Validation demonstrates computational consistency;
it does not establish causal identification or applicable legal wage floors.
