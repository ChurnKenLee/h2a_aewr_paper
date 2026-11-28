+++
title = "Narrow certification-space welfare calculation"
description = "Envelope-theorem certification-space producer-incidence bounds grid anchored to the FY2026 Q4 Track 1 lower bound, exposure and pass-through sensitivities, and paid-hours scaling."

[extra]
scopes = ["code/welfare", "scripts/run_welfare.sh"]
+++

The narrow welfare branch is a static, partial-equilibrium envelope-theorem
bounds calculator, not an additional causal estimator. Version 2
(`contract_version = 2`, `status = "scenario_bounds_grid"`, authorized
November 26, 2026) evaluates a declared certified-hours elasticity **bounds
grid** across the design-neutral [shared county-year
panel](@/contracts/shared-panel.md) (`3,108` contiguous U.S. counties,
`2011--2025`) rather than misrepresenting a noisy estimate as a single
precision-identified structural elasticity. Its contract
(`code/welfare/elasticity_contract.toml`) pins the shared panel, the Oct. 2,
2025 skill-rate table (`aewr_occupation_skill_rates.parquet`), the complete
FY2026 Q4 Track 1 statutory PPML certified-hours lower bound
(`outputs/tables/policy_2025/quantity_estimates.csv`), and the FY2026 Q4 R7
offered-wage pass-through (`outputs/tables/policy_2025/wage_estimates.csv`) by
exact SHA-256 digest and live CSV row parity.

{{ <grounding path="code/welfare/01_narrow_welfare.py" anchor="welfare-design-contract" sha256="15b51832f96bdabfdec63f4226e3ac13a7782f73b54d34e75caef75830fcd366" /> }}

The calculator fails closed if `status != "scenario_bounds_grid"`, if any
placeholder remains, if any pinned SHA-256 hash disagrees with disk, or if any
declared Track 1 or pass-through number disagrees with the published
`policy_2025` CSVs. Stage `01_narrow_welfare.py` (Marimo) evaluates the grid
and writes the provenance manifest; stage `02_validate_welfare.py` (Marimo)
independently verifies SHA-256 parity, balanced string keys, and analytic
envelope identities (`26/26` checks passed).

## Counterfactuals, envelope decomposition, and population

For baseline AEWR $a_0$, exposed certified hours $X_0$, effective pass-through
shift $\Delta_{\text{eff}} = \pi \Delta$, and elasticity
$\eta = d\log H^c / d\log a$, the calculator evaluates the exact envelope
integral $\Delta\Pi^{PE} = -\int_{a_0}^{a_0+\Delta_{\text{eff}}} X(s)\,ds$
under a constant-elasticity curve $H^c(a)=H^c_0(a/a_0)^\eta$ (with the exact
logarithmic limit $-X_0 a_0 \ln((a_0+\Delta_{\text{eff}})/a_0)$ at $\eta=-1$)
and checks functional-form sensitivity against a constant-semi-elasticity curve
$H^c(a)=H^c_0\exp((\eta/a_0)(a-a_0))$. Each scenario decomposes total producer
incidence into the exact first-order transfer $\Delta\Pi^{PE}(0) = -X_0
\Delta_{\text{eff}}$ and the nonnegative second-order behavioral cushion
$\Delta\Pi^{PE}(\eta) - \Delta\Pi^{PE}(0) \ge 0$ (loss mitigation for wage
increases; gain amplification for wage cuts), alongside the mechanical
certification-space H-2A wage-bill change $\Delta E^c_H$ (identically `0` at
$\eta = -1$ under constant elasticity).

The elasticity grid spans canonical benchmarks $\eta \in \{0.00, -0.30, -0.75,
-1.00\}$ plus the complete FY2026 Q4 Track 1 statutory PPML hours 95%
state-jackknife lower bound (**$\eta = -0.5985 \approx -0.60$**, `G = 46`
states; point estimate `-0.0778` retained strictly as a non-precision-identified
diagnostic). Every scenario is applied to all `3,108` shared-panel counties:

- **Primary `+$1.00/hr` counterfactuals**: `2011--2024` complete annual average
  (unweighted arithmetic mean of annual national totals) and `2011--2025`
  provisional-flagged average in 2012 farm-products-PPI real dollars
  (`aewr_ppi`), plus `2024` latest-complete-year snapshots in both PPI-2012
  real (`aewr_ppi`) and `2024` nominal (`aewr`) dollars.
- **Secondary finite policy-scale comparisons**: uniform `+-10%` real AEWR
  curvature stress tests (`2011--2024` and `2024` real) and the actual
  October 2, 2025 state-level U.S. skill-I cut relative to the `2025` FLS AEWR
  evaluated statically on `2024` certified hours (nominal USD).

## AEWR-exposed certified hours and pass-through sensitivity

The branch reports six exposure constructions for every primary window:

1. **Observed classified at AEWR** (`cert_hours_at_aewr_start_year`), which
   classifies explicit hourly offers within five cents of the contract
   start-year or prior-year AEWR (and from 2023-03-30 dated occupation or
   U.S.-worker skill-tier schedules; see the [AEWR input
   contract](@/contracts/aewr-inputs.md)). This is a wage-offer exposure proxy,
   not proof that the AEWR was the legally binding wage floor.
2. **Coverage-adjusted baseline**: applies each county-year's observed binding
   share among comparable hours to all certified hours, falling back to its
   AEWR-region-year and then national-year hour-weighted share when a
   positive-hours county-year lacks comparable offers (failing closed if no
   fallback exists).
3. **Parametric exposure bounds** $x \in \{0.85, 0.90, 0.94, 1.00\}$ applied to
   all certified hours (`man_hours_certified_start_year`), bracketing observed
   and institutional binding-share uncertainty.

Each primary and Oct. 2, 2025 scenario is reported both under statutory full
pass-through ($\pi = 1.00$) and under the FY2026 Q4 R7 offered-wage
pass-through sensitivity ($\pi = 0.6229$, SE `0.0996`).

## Headline certification-space bounds

All verified results come from `outputs/tables/welfare/welfare_bounds_grid.csv`
(`624` rows), `outputs/tables/welfare/welfare_annual_series.csv` (`300` rows),
`outputs/tables/table_welfare_bounds.tex`, and `outputs/reports/welfare_bounds.md`:

- **`2011--2024` average annual `+$1/hr` real (`aewr_ppi`, coverage-adjusted,
  `209.67M` exposed hrs/yr, $\pi = 1.00$)**: **`-$209.67M/yr`** at $\eta=0.00$
  (`0.45%` of `2011--2022` BEA `CAINC45` U.S. farm hired-labor expense and
  `0.081%` of crop cash receipts), **`-$204.99M/yr`** at the Track 1 Q4 95%
  lower bound ($\eta = -0.60$, second-order behavioral cushion **`+2.23%`**),
  and **`-$201.93M/yr`** at unit elasticity ($\eta = -1.00$, cushion
  **`+3.69%`**; constant-semi-elasticity differs by `0.097%`).
- **`2024` latest complete year `+$1/hr` nominal (`aewr`, hours-weighted mean
  `$16.65/hr`, `379.25M` exposed hrs, $\pi = 1.00$)**: **`-$379.25M`**
  ($\eta=0.00$), **`-$372.57M`** at Track 1 low ($\eta = -0.60$, cushion
  **`+1.76%`**), and **`-$368.18M`** at $\eta = -1.00$ (cushion **`+2.92%`**,
  $\Delta E^c_H = \$0.00\text{M}$). Parametric exposures span `-$333.6M` to
  `-$327.8M` (`x = 0.85`), `-$369.0M` to `-$362.5M` (`x = 0.94`), and
  `-$392.5M` to `-$385.6M` (`x = 1.00`) across $\eta \in [0, -0.60]$; applying
  the offered-wage pass-through ($\pi = 0.623$) scales `2024` nominal
  coverage-adjusted bounds to **`-$236.2M`** ($\eta=0.00$) through
  **`-$233.6M`** (Track 1 low, cushion `+1.11%`) to `-$231.9M` ($\eta = -1.00$).
- **Secondary finite policy-scale comparisons**: a uniform `+10%` real AEWR
  increase on `2024` hours yields `-$545.48M` ($\eta=0$) to `-$519.90M`
  ($\eta=-1$, `+4.69%` cushion). The actual October 2, 2025 U.S. skill-I cut
  (hours-weighted cut `-$3.74/hr` vs. `2025` FLS AEWR on static `2024` hours,
  nominal USD, $\pi = 1.00$) yields **`+$1,418.17M`** ($\eta=0.00$),
  **`+$1,533.05M`** at Track 1 low ($\eta = -0.60$, `+8.10%` gain
  amplification), and `+$1,618.0M` ($\eta = -1.00$).

Because the complete FY2026 Q4 Track 1 lower bound rules out elastic responses
below `-0.60`, the first-order envelope transfer captures over 97.7% of exact
partial-equilibrium producer incidence for a `+$1/hr` shift across the entire
admissible elasticity range.

## Interpretation and exclusions

The reported monetary object is certification-space farm quasi-rent:
producer incidence from the AEWR-exposed certified-hours curve, holding output
prices and other domestic-labor wages fixed. If paid hours are a constant
fraction $\rho$ of certified hours, paid-hours incidence is $\rho$ times the
reported result. The branch does not choose a value for $\rho$.

The calculation excludes corresponding-employment hours, H-2A worker surplus,
broader domestic labor-market effects, consumer incidence, administrative
costs, and general-equilibrium price responses. The `WPU01` farm-products PPI
(`aewr_ppi`) is treated as an agricultural output-price numeraire rather than a
general CPI/PCE household deflator and is paired with current-dollar `2024`
nominal snapshots.

Run from the repository root:

```sh
DRY_RUN=1 ./scripts/run_welfare.sh
./scripts/run_welfare.sh
```

Inputs, output paths, unit tests, and contract-pinning rules are listed in
`code/welfare/README.md`.
