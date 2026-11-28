# Full-panel FE-PPML bite design

Standalone full-panel county-year bite design (`2011–2025` estimation, `2008–2025`
event study) built directly on `data/processed/county_year_panel.parquet` without
modifying `code/c01_clean/`, `code/c02_build/`, `code/designs/did/`, or
`scripts/run_all.sh`.

## Run

From the repository root in the project devenv/renv environment:

```sh
DRY_RUN=1 ./scripts/run_bite.sh
./scripts/run_bite.sh
```

`scripts/run_bite.sh` sources `scripts/pipeline_helpers.sh`, runs R with
`--vanilla`, and executes the three stages in order.

## Execution order and ownership

| Stage | Script | Inputs | Outputs |
| --- | --- | --- | --- |
| 1 | `code/designs/bite/01_build_bite_panel.R` | `data/processed/county_year_panel.parquet` | `data/processed/bite_county_year_panel.parquet` |
| 2 | `code/designs/bite/02_estimate_bite.R` | `data/processed/bite_county_year_panel.parquet` | `outputs/tables/bite/bite_{main_estimates,diagnostics,event_study,loo_regions}.csv`, `outputs/tables/table_bite_{main,diagnostics}.tex`, `outputs/figures/fig_bite_event_study.{pdf,png}`, `outputs/reports/bite_design.md`, `outputs/reports/bite_provenance.json` |
| 3 | `code/designs/bite/03_validate_bite.R` | Shared panel, bite panel, provenance, and all retained outputs | Validation assertions (exits non-zero on any contract violation) |

Shared exposure, CV3J jackknife, effective-cluster ($G_{\text{eff}}$), variance-decomposition,
and LaTeX formatting helpers live in `code/designs/bite/helpers.R`.

## Estimand boundary and key conventions

- **Full-panel zeros**: Certified hours and positions are estimated with FE-PPML
  (`fixest::fepois`), avoiding `log(Y + 1)` / `asinh` distortions while retaining
  all switching counties cleanly.
- **Frozen `2008/2010` exposures**: `E_p25`, `E_oews`, and all-real 5-quantile
  shortfall `E_sf_log` are averaged strictly over `2008` and `2010` (preserving
  `AEWR-001`, where `2009` statutory AEWR is strictly NULL) and standardized
  within each of the 17 AEWR regions.
- **Relative differential estimand**: Coefficients on $E_c \times \ln(\text{AEWR}_{rt})$
  under `County + Region x Year FE` (`1b-PPML`) measure the relative log-point
  differential between high-bite and low-bite counties within the same AEWR
  region per $+1$ within-region SD of baseline exposure per unit increase in
  $\ln(\text{AEWR}_{rt})$. They are **not** aggregate AEWR elasticities.
- **Predetermined controls**: `+ X_pre x Year FE` interacts `2008–2010`
  predetermined covariates (`ln_pop_2010`, `farm_emp_share_0810`,
  `ln_emp_farm_0810`) with year indicators so `2023–2025` observations are not
  truncated by BEA `CAEMP25N` ending in `2022`.
- **Small-cluster inference**: Every specification reports both region-clustered
  CRV1 SEs and exact delete-one-cluster jackknife (**CV3J**) SEs with $t(G-1)$
  intervals and MDE $= 2.8 \times \text{CV3J SE}$.
- **Standalone domestic-floor margin**: The `ln(prevailing_ag_min_wage)` FE-PPML
  specification (`2011–2025`, all 48 contiguous states, state CRV1 and CV3J SEs)
  is reported as a standalone reduced-form domestic-floor margin and is never
  translated into an AEWR elasticity.
- **Provisional `2025` and `2008–2010` caveats**: `2025` start-year H-2A totals
  remain provisional pending FY2027 decisions, `2008–2010` weekly hours are
  pooled-mean imputed in the event study, and `nbr_employers_balanced_start_year`
  is split strictly at `2011–2019` vs `2020–2025` due to the FY2020 Addendum-B
  county-allocation structural break.

## Validate

```sh
bash -n scripts/run_bite.sh
DRY_RUN=1 ./scripts/run_bite.sh
Rscript --vanilla -e 'for (f in list.files("code/designs/bite", pattern = "\\.R$", full.names = TRUE)) parse(file = f)'
./scripts/run_bite.sh
```
