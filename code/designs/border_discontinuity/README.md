# AEWR-border redesign

This standalone branch compares the preserved v5 border-ring experiment with
four declared v6 local redesigns. The structural candidate screens every ring
county for predetermined agricultural relevance and requires locality to a
surviving qualifying border edge. Its empirical adequacy is assessed separately
from implementation; the branch does not select on significance.

The executable registry is `design.json`. See the
[canonical design page](../../../content/designs/border-discontinuity.md) for
estimands, source limitations, assumptions and inference. The generated finding
is `outputs/reports/border_redesign.md`.

## Run

From the repository root in the existing environment:

```sh
DRY_RUN=1 ./scripts/run_border_discontinuity.sh
./scripts/run_border_discontinuity.sh
```

To build and validate only the inputs:

```sh
H2A_BORDER_INPUTS_ONLY=1 ./scripts/run_border_discontinuity.sh
```

Input-only mode does not validate estimation or power. The branch remains
outside `run_all.sh`, the manuscript and welfare. Python entry points are
Marimo sources exported to `/tmp` by the runner. R entry points use
`Rscript --vanilla`; the helper selects the matching project library.
No dependency installation or source acquisition is part of this runner.

The separate input-upgrade runner reprocesses cached wage/Census inputs,
allocation/calendar exposure and geocoding requests, rebuilds the shared panel
and its affected design consumers, then runs border stages09--11:

```sh
DRY_RUN=1 ./scripts/run_border_data_inputs.sh
./scripts/run_border_data_inputs.sh
```

Public geocoding acquisition requires `H2A_RUN_PUBLIC_GEOCODER=1`; default runs
only prepare requests/read caches. Non-public NASS data and custom tabulations
are unavailable; the former access proposal is withdrawn. The county analysis
uses public BEA/Census denominators and needs no institutional-access details.
The DOL archival draft is never sent by this runner. Source caches and reviewed
employer linkage must already exist. The bounded MC queue remains the default.

## Execution and ownership

| Order / script | Inputs and owned outputs |
| --- | --- |
| `01_build_county_features.py` | Immutable shared panel, crop/climate/soil and 2010 land area → branch county features including frozen agricultural support |
| `02_construct_geographies.py` | Features and immutable adjacency/region mapping → six membership specifications, geography diagnostics, assignment exclusions, agricultural border audit |
| `03_build_panel.py` | Shared panel, branch features/memberships → unique county-year panel, fixed baseline history and preemployment weights |
| `04_estimate_models.R` | Panel/memberships → original seven v5 models, annual/balance/placebo/support/leave-pair-out and exact county/stratum/region/pair decomposition audit |
| `05_generate_diagnostics.R` | Benchmark and geography artifacts → retained v5 tables, maps and diagnostic figures |
| `07_estimate_redesign.R` | Panel/memberships/registry → eight comparable original/new policy fits, support, balance, earlier-wage diagnostics, region deletion, FWL, common-sample/noise/assignment audits and simulation-input RDS |
| `08_power_and_report.R` | Simulation inputs and new results → conditional size/power, calibration, signed MDE, scalar bootstrap simulation, negative-variance diagnostics, precision figure, report and provenance JSON |
| `06_validate_artifacts.py` | All branch artifacts → schema, geography, registry, metadata, numerical and simulation checks; no persistent output |
| `09_compare_data_inputs.R` | `data_input_design.json`, shared/BEA/Census/calendar inputs → four fixed-denominator native/common comparisons and bounded common-sample power |
| `10_report_data_inputs.py` | Source manifests, comparisons and source-year coverage → input-upgrade report, public-data plan/DOL archival draft, audit CSVs and report manifest |
| `11_validate_data_inputs.py` | Input-upgrade artifacts → independent accounting, missingness, registry, simulation and provenance checks in a validation JSON |
| `12_construct_statewide_geography.py` | `statewide_design.json`, features, adjacency → `border_statewide_memberships.parquet` and `border_statewide_pair_audit.parquet` (state-border county pairs) |
| `13_statewide_power.R` | Memberships, AEWR schedules, pre-period FY counts → `border_statewide_power_panel.parquet`, `border_statewide_{treatment_support,calibration,power,mde,decision}.csv`, `outputs/reports/border_statewide_power.md` |

`geography.py` owns the design's geographic algorithms; `redesign.R`,
`power.R` and `bootstrap_power.R` contain its numerical helpers. The shared
normalizers, path helpers, source producers and common inputs remain upstream.
Stages 12--13 implement the separately registered statewide-AEWR design
(`statewide_design.json`); they reuse `geography.py` and `power.R` and read
no post-policy outcome. The diagnostic business margin remains reviewed balanced v2 worksite-business
counts, with the inherited FY2024 evidence and review-report provenance.

## Principal artifacts

The original `border_discontinuity_*` feature/membership/panel and
`border_discontinuity_ring_*` credibility artifacts remain benchmarks.
New retained outputs are:

- `outputs/tables/border_discontinuity_redesign_estimates.csv`
- `outputs/tables/border_discontinuity_redesign_sample_audit.csv`
- `outputs/tables/border_discontinuity_redesign_support.csv`
- `outputs/tables/border_discontinuity_redesign_region_deletion.csv`
- `outputs/tables/border_discontinuity_redesign_balance.csv`
- `outputs/tables/border_discontinuity_redesign_placebo.csv`
- `outputs/tables/border_discontinuity_redesign_fwl.csv`
- `outputs/tables/border_discontinuity_redesign_comparable_samples.csv`
- `outputs/tables/border_discontinuity_redesign_noise_diagnosis.csv`
- `outputs/tables/border_discontinuity_redesign_assignment_audit.csv`
- `outputs/tables/border_discontinuity_redesign_power.csv`
- `outputs/tables/border_discontinuity_redesign_power_calibration.csv`
- `outputs/tables/border_discontinuity_redesign_mde.csv`
- `outputs/tables/border_discontinuity_redesign_simulation_diagnostics.csv`
- `outputs/tables/border_discontinuity_redesign_bootstrap_power.csv`
- `outputs/tables/border_discontinuity_redesign_decision.csv`
- `outputs/figures/border_discontinuity_redesign_precision.png`
- `outputs/reports/border_redesign.md`
- `outputs/reports/border_discontinuity_redesign_provenance.json`
- `data/intermediate/border_discontinuity_redesign_simulation_inputs.rds`

The upstream correction makes canonical 2009 wages null. Stage04 alone uses
`v5_aewr_ppi_comparator` to reproduce the contaminated historical audit.
Stage07 uses canonical wages, 2008/2010 wage diagnostics and all three outcome
years for FE-only noise calibration. The 2011--2022 treatment is unchanged.

Stage09 owns `data/intermediate/border_data_input_denominators.parquet`, the
unique `data/processed/border_data_input_county_year.parquet`, and the
`border_data_input_{estimates,sample_audit,power,calibration,mde,simulation_diagnostics}.csv`
tables. It owns `outputs/reports/border_data_input_provenance.json`.
Stage10 owns `border_data_inputs.md`, `border_data_access_requests.md` (the
public-data plan and unsent DOL archival draft), and
`border_data_input_report_manifest.json` under `outputs/reports`, plus the
source-coverage, worksite-date, unreconciled-case and access-variable CSVs.
Stage11 owns `outputs/reports/border_data_inputs_validation.json`.
The [input-comparison contract](../../../content/designs/border-data-inputs.md)
defines denominator units, coverage, common samples and external requirements.

## Focused checks

The test modules use plain functions and do not require pytest:

```sh
UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync python - <<'PY'
import runpy
for name in ('test_geography', 'test_redesign_validation'):
    scope = runpy.run_path(f'code/designs/border_discontinuity/tests/{name}.py')
    for key, value in scope.items():
        if key.startswith('test_') and callable(value):
            value()
PY
```

Run each R numerical test using the same library helper, for example:

```sh
bash -c 'source scripts/pipeline_helpers.sh; run_step code/designs/border_discontinuity/tests/test_redesign.R'
bash -c 'source scripts/pipeline_helpers.sh; run_step code/designs/border_discontinuity/tests/test_power.R'
bash -c 'source scripts/pipeline_helpers.sh; run_step code/designs/border_discontinuity/tests/test_bootstrap_power.R'
```

The full branch run is the retained-artifact check. Run the ordinary branch
after any shared-panel rebuild and stages09--11 after calendar or denominator
input changes. Grounding verification and Zola checks remain required after
documentation changes.
