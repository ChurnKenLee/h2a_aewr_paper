+++
title = "Artifact ownership"
description = "Which pipeline stage owns each persistent artifact."

[extra]
scopes = ["code", "scripts"]
+++

Each persistent artifact has one owning producer. Schema corrections and
transformations belong in that producer rather than in downstream repair code.

| Stage | Responsibility | Principal artifact class |
| --- | --- | --- |
| `code/a01_sources` | Acquire and normalize external sources | Source Parquets |
| `code/b01_derived` | Construct reusable measures and prediction inputs | Derived Parquets |
| `code/c01_clean` | Normalize source families and merge the county-year backbone | `county_year_merged.parquet` |
| `code/c02_build` | Construct the design-neutral shared panel | `county_year_panel.parquet` |
| `code/descriptives` | Produce design-neutral figures | Retained figures |
| `code/designs/*` | Construct treatments, instruments, samples, and estimators | Design panels and results |
| `code/welfare` | Evaluate the Track-1-anchored certification-space envelope bounds grid | Certification-space bounds panel, CSVs, table, report, and manifest |

{{ <grounding path="scripts/run_all.sh" anchor="pipeline-order" sha256="6ce943803a89bf2ce24018b67f7d970cdafdaa846a5ecf3eed14c0c64d8ecccf" /> }}

Storage locations are part of the contract:

- `data/raw` contains source files and hand-maintained crosswalks.
- `data/intermediate` contains exchange artifacts owned by one producer.
- `data/processed` contains analysis-ready panels.
- `outputs/figures` and `outputs/tables` contain retained empirical products.
- `outputs/reports` contains generated methodological reports and provenance.

The shared stages remain design-neutral. Treatment definitions, post
indicators, fixed-effect factors, target clusters, and instruments are owned by
their design branches.

The A01 H-2A matcher owns the v2 worksite-business crosswalk, source-role
membership, edge and cluster audits, review sample/report, and the full-vintage
versus as-of-2022 comparison. B01 consumes the reviewed membership to construct
positive-certified county-year business counts; C01 and C02 only validate and
propagate those counts and linkage provenance.

A01's AEWR producer owns annual references, dated FLS and occupation/skill
lookups, the nonstatutory 2009 archive, policy ledger and source provenance.
A01's `07_state_minimum_wages.py` owns `state_year_min_wage.parquet` (unified
FRED release 387 across all 51 states/DC and federal annual years, retiring
`code/utilities/pull_fred_minimum_wages.R`), and C01's `04_state_minimum_wages.R`
owns `state_real_minwages.parquet` with dated agricultural coverage
(`state_ag_minwage_coverage.csv`).
B01/01 owns case/allocation source audits; B01/01b owns the monthly certification
calendar, county-year capacity, allocation/coverage audits and manifest.
A01/02_05 owns actual worksite address records, normalized address IDs, public
geocoding requests/responses and acquisition metadata. It does not infer
spatial-RD readiness. Their contracts distinguish source facts, allocation
assumptions and unresolved access.

The panel-IV branch owns one original moving-wage pipeline in R. Stage03 owns
the rolling frame and its diagnostics;04 owns deterministic county features,
weights and calibration/moment diagnostics;05 owns donor accounting and
cluster-year instruments;06 owns `panel_iv_county_year.parquet`;07 owns the
retained estimates/tables and email exports;08 owns diagnostic figures and
plotting-data CSVs;09 validates without writing another copy;10 owns the
instrument-decomposition parquet and its support and variance tables;11 owns
the QWI wage-granularity diagnostic tables and writes no instrument artifact.
A01 `04_04` owns the QWI earnings input of stage11. Experimental revision
outputs and simulated draw partitions
have been deleted by user instruction. Shared inputs and other designs remain
outside this retirement.

The branch also owns the explicit Census farm-floor source-row selection and
its source fingerprint. Stage 03 validates this input against local QuickStats
before rebuilding; refreshing it requires a review of the prior definition.

The AEWR-border branch owns its feature/membership/panel and retained v5 audit.
Stage07 owns `border_discontinuity_redesign_*` estimates, support, assignment,
FWL, deletion, comparable-sample and noise evidence plus simulation-input RDS.
Stage08 owns signed power/MDE/calibration/decision outputs, precision figure,
`outputs/reports/border_redesign.md` and its provenance JSON;06 validates both
registries. Exact filenames are listed in its README. Shared feature helpers
under `src/h2a`, path/geography helpers and source panels remain upstream inputs.
New county-year analysis panels retain unique string county/year keys; expanded
border memberships have their own declared key and duplication weights.

Border09 separately owns four denominator inputs, the unique county-year
comparison panel, native/common estimates and bounded power diagnostics.
Border10 owns the input-upgrade report, source/year and date/case audits,
public-data plan, unsent DOL archival draft and its variable-concept CSV.
Non-public NASS access/tabulations are excluded by user instruction. Border11 owns the
independent input-validation JSON. No candidate replaces the benchmark
denominator or moves design-specific normalization into C01/C02.

`code/welfare` is a standalone downstream bounds consumer rather than an
estimator. Its two Marimo stages (`01_narrow_welfare.py`,
`02_validate_welfare.py`, backed by `envelope.py`) read the shared panel,
`aewr_occupation_skill_rates.parquet`, and the SHA-256-pinned
`elasticity_contract.toml` anchored to the FY2026 Q4 `policy_2025` Track 1
lower bound and offered-wage pass-through, and own
`welfare_certification_space_county_year.parquet`,
`outputs/tables/welfare/{welfare_bounds_grid,welfare_annual_series}.csv`,
`outputs/tables/table_welfare_bounds.tex`, `outputs/reports/welfare_bounds.md`,
and `data/intermediate/welfare_certification_space_manifest.json`. It does not
alter the shared panel or select an empirical specification.

The historical wage-reform branch owns three `wage_reforms_*` design Parquets
for positive cases, all application case-years and fixed business cohorts.
Stage01 owns those panels and coverage/geography/timing/linkage CSVs; stage02
owns local wage, cohort response and state-deletion CSVs; stage03 owns
`outputs/reports/wage_reforms.md`, PNG/PDF figures and provenance; stage04
validates them. These source-level diagnostics do not replace B01's county
aggregates or allocate joint-case positions to individual businesses.

The separate `code/designs/policy_2025` branch owns the current 2025 wage-offer,
Track 1 county H-2A quantity, mechanism-heterogeneity, and interim
covered-earnings work. Its inputs stages own the design-specific filer
reconstruction; 01 owns the occupation/certification panels and wage audits;
02 owns wage fits; 05/09 own earnings extracts; 06/10 own pre-policy and
interim fits; 11 owns the complete FY2026 Q4 Track 1 county quantity panel
(`quantity_county_fy.parquet`), 79 quantity fits, cutoff/cost-conversion CSVs,
and `outputs/tables/table_policy_2025_quantity.tex`; 11b owns the five-panel
mechanism-heterogeneity panel (`mechanisms_county_fy.parquet`), 96 mechanism
fits (12 primary Holm-adjusted tests), mechanism CSVs,
`outputs/tables/table_policy_2025_mechanisms.tex`, and
`outputs/figures/fig_policy_2025_mechanisms.{pdf,png}`; 07 validates all 703
rows; 08 generates the integrated report, complete ledger,
`outputs/tables/table_policy_2025_{wages,earnings}.tex`, and
`outputs/figures/fig_policy_2025_wages.{pdf,png}`; 12 publishes all 52
validated artifacts and their hashes. Filer exchange artifacts live under
`data/intermediate/policy_2025`, analysis panels under
`data/processed/policy_2025`, CSVs and publication LaTeX tables under
`outputs/tables/`, publication figures under `outputs/figures/`, and
reports/provenance under `outputs/reports`. Earlier exploratory code and
reports remain historical evidence. The original exposure file is retained
verbatim as a frozen design input, not reconstructed from later outcomes.

The standalone `code/designs/bite` branch (`scripts/run_bite.sh`, outside
`scripts/run_all.sh`) owns the full-panel FE-PPML bite design without modifying
shared C01/C02 or retained DiD artifacts. Stage `01_build_bite_panel.R` owns
`data/processed/bite_county_year_panel.parquet`; stage `02_estimate_bite.R` owns
`outputs/tables/bite/bite_{main_estimates,diagnostics,event_study,loo_regions}.csv`,
`outputs/tables/table_bite_{main,diagnostics}.tex`,
`outputs/figures/fig_bite_event_study.{pdf,png}`, `outputs/reports/bite_design.md`,
and `outputs/reports/bite_provenance.json`; stage `03_validate_bite.R` validates
those artifacts and their SHA-256 parity against `county_year_panel.parquet`.
