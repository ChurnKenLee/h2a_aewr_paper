# Border-discontinuity design (future)

This directory reserves the design boundary for a future border-discontinuity
analysis. No implementation is supported yet.

Future scripts must consume `data/processed/county_year_panel.parquet`, keep
border-specific treatment and running-variable construction inside this
directory, prefix design intermediates with `border_discontinuity_`, and add a
dedicated runner without changing the shared-panel contract.
