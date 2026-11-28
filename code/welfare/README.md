# Narrow certification-space welfare bounds

This directory owns the static, partial-equilibrium certification-space
producer-incidence bounds calculation under the envelope theorem
($\Delta\Pi^{PE} = -\int X(s)\,ds$). It is deliberately separate from the
empirical estimators and remains outside `scripts/run_all.sh`.

## Run

From the repository root:

```sh
DRY_RUN=1 ./scripts/run_welfare.sh
./scripts/run_welfare.sh
```

`scripts/run_welfare.sh` executes two imperative Marimo stages via `run_step`:

| Stage | Script | Responsibility |
| --- | --- | --- |
| 1 | `code/welfare/01_narrow_welfare.py` | Validate `elasticity_contract.toml` hashes, evaluate the full elasticity/exposure/pass-through bounds grid, write outputs and provenance manifest |
| 2 | `code/welfare/02_validate_welfare.py` | Independently verify SHA-256 parity, balanced string keys, and exact analytic envelope identities |

Shared pure-Python math and contract helpers live in `code/welfare/envelope.py`
and are covered by `code/welfare/tests/test_welfare.py` (run with
`UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync python -m unittest discover -s code/welfare/tests`).

## Contract and outputs

`elasticity_contract.toml` (`contract_version = 2`,
`status = "scenario_bounds_grid"`) pins `data/processed/county_year_panel.parquet`,
`data/intermediate/aewr_occupation_skill_rates.parquet`, the complete FY2026 Q4
Track 1 quantity lower bound (`outputs/tables/policy_2025/quantity_estimates.csv`),
and the FY2026 Q4 offered-wage pass-through (`outputs/tables/policy_2025/wage_estimates.csv`)
by SHA-256. The calculator fails closed if any hash disagrees with disk.

Outputs:

- `data/processed/welfare_certification_space_county_year.parquet`
- `outputs/tables/welfare/welfare_bounds_grid.csv`
- `outputs/tables/welfare/welfare_annual_series.csv`
- `outputs/tables/table_welfare_bounds.tex`
- `outputs/reports/welfare_bounds.md`
- `data/intermediate/welfare_certification_space_manifest.json`

## Estimand boundary

- The elasticity grid (`0.0, -0.3, -0.75, -1.0`, plus the Track 1 Q4 jackknife
  95% CI lower bound and diagnostic point estimate) is explicitly a **bounds
  grid**, not a single precision-identified point estimate.
- Primary counterfactuals evaluate an anticipated `+$1.00/hr` AEWR increase over
  `2011--2024` (complete), `2011--2025` (provisional `2025` flagged), and `2024`
  in both PPI-2012 real (`aewr_ppi`) and `2024` nominal (`aewr`) dollars.
  Secondary comparisons evaluate uniform `+-10%` real AEWR shifts and the actual
  Oct. 2, 2025 U.S. skill-I cut on static `2024` certified hours.
- All monetary outputs are in **certification space**. If realized paid hours
  equal a constant fraction `rho` of certified hours, paid-hours incidence
  equals `rho` times every reported monetary value.
- Output prices, domestic wages, corresponding-employment hours, worker surplus,
  and general-equilibrium adjustments are held fixed and excluded.
