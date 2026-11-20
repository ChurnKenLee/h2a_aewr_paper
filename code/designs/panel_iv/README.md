# Panel IV

This branch estimates the original moving-wage panel IV from the design-neutral
shared panel. All stages, numerical helpers, and tests are R. The fixed-wage
revision, the subsequent separation/innovation exercises, the wage-only
instrument and the `continuing` control set have been retired and their
generated artifacts deleted at the user's request. Stages 10--11 are
diagnostics, a Kitagawa decomposition of the instrument and its sensitivity to
county-level QWI donor wages; they change no instrument, column or table.

## Run

From the repository root in the configured direnv/devenv environment:

```sh
DRY_RUN=1 ./scripts/run_panel_iv.sh
./scripts/run_panel_iv.sh
```

The runner uses `Rscript --vanilla` and the existing project R library. It reads
local inputs and does not acquire source data. Run the shared-panel producer
first when its inputs have changed. Stage 11 also reads
`qwi_county_ag_quarterly_earnings.parquet` from
`code/a01_sources/04_04_qwi_quarterly_earnings.py`.

## Order and ownership

| Stage | Principal outputs |
| --- | --- |
| `01_build_county_features.R` | County clustering features |
| `02_cluster_target_units.R` | Fixed target/donor clusters and map |
| `03_build_fls_frame.R` | `panel_iv_fls_frame.parquet` and frame diagnostics |
| `04_recover_fls_geography.R` | County features, deterministic weights, calibration and moment diagnostics |
| `05_construct_instruments.R` | County donor frame and cluster-year instruments |
| `06_build_county_year_panel.R` | `data/processed/panel_iv_county_year.parquet` |
| `07_estimate_panel_iv.R` | Two-column first stage and twelve-outcome 2SLS tables for policy years 2011--2023, the static-trend diagnostic, samples, summary statistics and email exports |
| `08_generate_figures.R` | Six retained diagnostic figures and plotting-data CSVs |
| `09_validate_artifacts.R` | Source, calibration, instrument, sample and retained-output checks |
| `10_decompose_instrument.R` | `panel_iv_instrument_decomposition_cluster_year.parquet` and `iv_instrument_decomposition_{support,variance}.csv` |
| `11_qwi_wage_diagnostics.R` | `iv_qwi_wage_diagnostics_{coverage,summary,first_stage}.csv`: QWI coverage, variance shares and first stages for the design and for the OEWS wage times an annual county QWI factor with weights recalibrated on that wage |

`frame.R` owns the rolling Census/QCEW/QWI/BEA prior. `calibration.R` owns moment
construction and deterministic entropy recovery. `io.R` provides atomic writes
and partial-rebuild schema guards. `validation.R` checks retained artifacts.
`estimation_spec.json`, read by `specification.R`, is the single model/outcome
registry. `design.R` owns the target/donor and calibration constants.
`decomposition.R` owns the decomposition constants, the base wage, the
balanced-donor rule and the Kitagawa identity, shared by stages 10 and 11.

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

The single instrument is `z_wage_seasonal_composition_real`; stage 04 recovers
only its wage-plus-seasonal/composition county distribution. Column 1 has
no controls and column 2 adds the controls; column 2 is preferred. Both columns
share the controlled specification's complete-case sample, which ends in policy
year 2023 because the BEA employment controls end in 2022. The original input
construction and samples are preserved, including reviewed v2
linked-worksite-business counts.

## Validation

Run the five `tests/test_*.R` files with `run_step` or configured
`Rscript --vanilla`. They cover calibration, frame completeness, geographic
keys, atomic writes, partial rebuilds, outcome definitions, donor wages, the
balanced-donor rule and the decomposition identity.
Stage 09 independently checks artifacts. No Python or project C adapter is
needed by this branch.

See the [canonical design](../../../content/designs/panel-iv.md) for the
estimand, moment definitions, source limitations and retained products.
