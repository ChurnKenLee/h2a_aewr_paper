# Panel IV

This branch estimates the original moving-wage panel IV from the design-neutral
shared panel. All stages, numerical helpers, and tests are R. The fixed-wage
revision and subsequent separation/innovation exercises have been retired and
their generated artifacts deleted at the user's request. Stages 10--11 add a
user-requested Kitagawa decomposition diagnostic of the retained instruments;
they do not change any retained instrument, column or table.

## Run

From the repository root in the configured direnv/devenv environment:

```sh
DRY_RUN=1 ./scripts/run_panel_iv.sh
./scripts/run_panel_iv.sh
```

The runner uses `Rscript --vanilla` and the existing project R library. It reads
local inputs and does not acquire source data. Run the shared-panel producer
first when its inputs have changed.

## Order and ownership

| Stage | Principal outputs |
| --- | --- |
| `01_build_county_features.R` | County clustering features |
| `02_cluster_target_units.R` | Fixed target/donor clusters and map |
| `03_build_fls_frame.R` | `panel_iv_fls_frame.parquet` and frame diagnostics |
| `04_recover_fls_geography.R` | County features, deterministic weights, calibration and moment diagnostics |
| `05_construct_instruments.R` | County donor frame and cluster-year instruments |
| `06_build_county_year_panel.R` | `data/processed/panel_iv_county_year.parquet` |
| `07_estimate_panel_iv.R` | Four-column first stages and twelve-outcome 2SLS tables, samples, summary statistics and email exports |
| `08_generate_figures.R` | Six retained diagnostic figures and plotting-data CSVs |
| `09_validate_artifacts.R` | Source, calibration, instrument, sample and retained-output checks |
| `10_decompose_instruments.R` | `panel_iv_instrument_decomposition_cluster_year.parquet` and `iv_instrument_decomposition_support.csv` |
| `11_estimate_instrument_decomposition.R` | `iv_instrument_decomposition_{first_stage,variance,second_stage,overidentification}.csv` and two `table_iv_instrument_decomposition_*.tex` tables |

`frame.R` owns the rolling Census/QCEW/QWI/BEA prior. `calibration.R` owns moment
construction and deterministic entropy recovery. `io.R` provides atomic writes
and partial-rebuild schema guards. `validation.R` checks retained artifacts.
`estimation_spec.json`, read by `specification.R`, is the single model/outcome
registry. `design.R` owns the target/donor and calibration constants.
`decomposition.R` owns the decomposition constants, the Kitagawa identity, the
Anderson-Rubin set and the cluster-robust Hansen J.

`census_farm_floor_selection.csv` records the original source-row choices for
the all-domain Census farm floor. `census_farm_floor_source.json` fingerprints
the local QuickStats input. Stage 03 verifies both before use and fails if the
source changes; review the selection before rebuilding with refreshed data.
These explicit choices preserve the former Polars tie ordering without a
Python dependency. They are not a corrected TOTAL-domain farm count.

Stage 04 accepts `--years 2010 2011` and `--regions 1 2` for partial rebuilds.
Use the runner's existing helper so the configured R library is available:

```sh
source scripts/pipeline_helpers.sh
run_step code/designs/panel_iv/04_recover_fls_geography.R --years 2010 2011 --regions 1 2
```

A complete stage 03--09 rebuild is required when migrating from the former
simulation-bearing artifacts. Stage 04 writes `calibration_version =
panel_iv_deterministic_v1`; a partial rebuild rejects incompatible retained
cells before writing. There are no draw partitions, simulation envelopes,
random-stream adapters, or simulated-draw success guards. Deterministic
convergence, normalization, moment-fit and coverage checks remain mandatory.
After changing the frame or calibration, rebuild every downstream stage.

The retained instruments are `z_wage_only_real` and
`z_wage_seasonal_composition_real`. Columns 1--4 are wage-only,
seasonal/composition, wage-only with controls, and seasonal/composition with
controls; column 4 remains preferred. The original input construction and
samples are preserved, including reviewed v2 linked-worksite-business counts.

## Validation

Run the five `tests/test_*.R` files with `run_step` or configured
`Rscript --vanilla`. They cover calibration, frame completeness, geographic
keys, atomic writes, partial rebuilds, outcome definitions, donor wages and
the decomposition identity, AR sets and Hansen J.
Stage 09 independently checks artifacts. No Python or project C adapter is
needed by this branch.

See the [canonical design](../../../content/designs/panel-iv.md) for the
estimand, moment definitions, source limitations and retained products.
