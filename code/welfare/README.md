# Narrow welfare calculation

This directory owns the static, partial-equilibrium certification-space
producer-incidence calculation for an anticipated real one-dollar AEWR
increase. It is deliberately separate from the empirical designs that estimate
the certified-hours elasticity.

Run from the repository root:

```sh
./scripts/run_welfare.sh
```

The runner stops while `elasticity_contract.toml` has `status = "unfrozen"`.
After the empirical procedure selects one credible elasticity, freeze the
contract with its estimate, standard error, exact source artifact and hash,
source-row selector, estimand population, target years, and real-AEWR units.

The calculation consumes `data/processed/county_year_panel.parquet`, applies
the frozen elasticity to all shared-panel counties in the declared target
years, and writes:

- `data/processed/welfare_certification_space_county_year.parquet`;
- `outputs/tables/welfare_certification_space_annual.csv`;
- `outputs/tables/welfare_certification_space_summary.csv`;
- `outputs/tables/table_welfare_certification_space.tex`; and
- `data/intermediate/welfare_certification_space_manifest.json`.

The headline is the unweighted mean of annual national totals, not the sum
across years. The observed result uses certified hours explicitly classified at
the current or lagged AEWR. The coverage-adjusted sensitivity extrapolates the
observed binding share using county-year information, then AEWR-region-year and
national-year fallbacks.

All monetary results are in certification space. If paid hours are a constant
fraction `rho` of certified hours, the corresponding paid-hours result is
`rho` times the reported value. The calculation does not include corresponding
employment, worker surplus, domestic labor-market effects, consumer incidence,
or general-equilibrium price responses.
