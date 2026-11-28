# Certification-Space Producer Incidence Bounds (Path 1 Envelope Grid)

- **Contract**: `code/welfare/elasticity_contract.toml` (`contract_version = 2`, `status = "scenario_bounds_grid"`, authorized `2026-11-26`).
- **Shared panel**: `data/processed/county_year_panel.parquet` (SHA-256 `4e2e56d0884715ea0ff26b8251339c18748cc1966692a5ab0eac31171cf51b92`).
- **Track 1 elasticity anchor**: `outputs/tables/policy_2025/quantity_estimates.csv` (SHA-256 `dd4568251fa2de7e24f7812a110084cb32ef700dff2a81bc9a5f0548c2766d5c`; `window=full_fy,contrast=fy2025_2026,specification=B_primary,outcome=hours,estimator=PPML,treatment=x_us_skill_1`): point `-0.0778` (diagnostic only), CV3J SE `0.2585`, 95% CI `[-0.5985, 0.4428]` across `46` states.
- **Offered-wage pass-through anchor**: `outputs/tables/policy_2025/wage_estimates.csv` (SHA-256 `d1fe06fc43b94bc59e6272d0e7227ebcd0952d5545f7799028e8a18825268d30`; `track=wage,specification=occupation_dw,tau=2026,tier=R7,weights=h0,term=dose`): `pi = 0.6229` (SE `0.0996`).
- **Paid-hours scaling**: Every monetary figure is in **certification space**. If realized paid hours equal a constant fraction `rho` of certified hours, paid-hours producer incidence equals `rho` times the reported amount.

## Headline Envelope Bounds

1. **2011–2024 average annual `+$1` real (`aewr_ppi`) increase (coverage-adjusted, `pi = 1.00`)**:
   - First-order transfer (`eta = 0.00`): `$-209.67M`/yr (`209.67M` exposed hrs).
   - Unit-elastic benchmark (`eta = -1.00`): `$-201.93M`/yr (2nd-order behavioral cushion `+3.69%`; constant-semi-elasticity differs by only `0.097%`).
   - Complete FY2026 Q4 Track 1 lower bound (`eta = -0.60`): `$-204.99M`/yr (2nd-order cushion `+2.23%`).
   - BEA `CAINC45` scale (`2011–2022` real): `0.45%` of U.S. farm hired-labor expense (`0.081%` of crop cash receipts) at `eta = 0`.
2. **2024 latest complete year `+$1` nominal (`aewr`) increase (coverage-adjusted, `pi = 1.00`)**:
   - `eta = 0.00`: `$-379.25M` (`379.25M` exposed hrs, hours-weighted baseline AEWR `$16.65`/hr).
   - `eta = -1.00`: `$-368.18M` (cushion `+2.92%`, H-2A wage-bill change `$-0.00M`).
   - Track 1 Q4 lower bound (`eta = -0.60`): `$-372.57M` (cushion `+1.76%`).
   - Offered-wage pass-through sensitivity (`pi = 0.623`, `eta = -1.00`): `$-231.87M`.
3. **Secondary finite policy-scale comparisons**:
   - Uniform `+10%` real AEWR on 2024 hours: `$-545.48M` (`eta = 0`) vs `$-519.90M` (`eta = -1.00`, cushion `+4.69%`).
   - Actual Oct 2, 2025 state-level U.S. skill-I cut vs 2025 FLS AEWR (mean weighted cut `$-3.74`/hr, static 2024 hours, nominal USD, `pi = 1.00`): `$+1,418.17M` (`eta = 0`) to `$+1,533.05M` (Track 1 lower bound, cushion `+8.10%`).

## Annual National Totals (`2011–2025`, `+$1` Real AEWR, Coverage-Adjusted Exposure)

| Year | Provisional | Baseline Hours (M) | Exposed Hours (M) | `eta=0.00` ΔΠ ($M) | `eta=-1.00` ΔΠ ($M) | Track 1 Low ΔΠ ($M) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2011 | No | 86.00 | 81.13 | -81.13 | -77.30 | -78.81 |
| 2012 | No | 90.75 | 83.55 | -83.55 | -79.57 | -81.14 |
| 2013 | No | 100.01 | 94.24 | -94.24 | -89.80 | -91.54 |
| 2014 | No | 116.39 | 110.71 | -110.71 | -105.61 | -107.62 |
| 2015 | No | 137.72 | 132.75 | -132.75 | -127.44 | -129.54 |
| 2016 | No | 157.83 | 151.74 | -151.74 | -146.47 | -148.56 |
| 2017 | No | 188.42 | 181.83 | -181.83 | -175.52 | -178.02 |
| 2018 | No | 224.05 | 215.68 | -215.68 | -208.42 | -211.30 |
| 2019 | No | 245.15 | 235.94 | -235.94 | -228.36 | -231.37 |
| 2020 | No | 263.96 | 254.09 | -254.09 | -246.48 | -249.50 |
| 2021 | No | 307.96 | 295.72 | -295.72 | -285.10 | -289.30 |
| 2022 | No | 362.30 | 349.25 | -349.25 | -334.37 | -340.24 |
| 2023 | No | 382.03 | 369.43 | -369.43 | -356.03 | -361.33 |
| 2024 | No | 392.52 | 379.25 | -379.25 | -366.51 | -371.56 |
| 2025 | Yes | 399.46 | 381.97 | -381.97 | -368.87 | -374.06 |

## Maintained Partial-Equilibrium Exclusions

- Holds farm-gate/retail output prices and non-H-2A domestic wages fixed (`dΠ/da = -X(a)`).
- Excludes corresponding-employment domestic hours, H-2A foreign worker outside-option surplus, administrative fee costs, and long-run crop/mechanization adjustments.
- Treats the `WPU01` farm-products PPI (`aewr_ppi`) as an agricultural output-price numeraire rather than a general CPI/PCE household deflator, paired with nominal `2024` current-dollar snapshots.
