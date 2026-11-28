+++
title = "Full-panel FE-PPML bite design"
description = "Standalone full-panel county-year FE-PPML bite estimand, frozen 2008/2010 exposures, CV3J jackknife inference, diagnostics, and standalone domestic-floor margin."

[extra]
scopes = ["code/designs/bite", "scripts/run_bite.sh", "outputs/reports/bite_design.md"]
+++

`code/designs/bite` implements a standalone full-panel county-year FE-PPML bite
design over the unified [shared county-year panel](@/contracts/shared-panel.md)
(`3,108` contiguous U.S. counties across `48` states and `17` AEWR regions,
`2008--2025`, `55,944` county-years). Its runner `scripts/run_bite.sh` remains
outside `scripts/run_all.sh` and leaves `code/c01_clean/`, `code/c02_build/`,
and the [retained DiD branch](@/designs/difference-in-differences.md) untouched.

{{ <grounding path="scripts/run_bite.sh" anchor="bite-runner" sha256="7e9b42b45cef866a2fc08b9b8e5e6244c82eec8a4d034872687ec4e5ed6abc9e" /> }}

Run from the repository root:

```sh
DRY_RUN=1 ./scripts/run_bite.sh
./scripts/run_bite.sh
```

## Frozen exposures and estimand boundary

Unlike the retained DiD branch—which drops `"always takers"` (62.6% of U.S.
H-2A workers), uses a time-varying local wage gap (`aewr_cz_p25_l1`), and loses
`2023--2025` under time-varying BEA controls—this branch retains the full
county panel, uses fixed-effects Poisson pseudo-maximum likelihood
(`fixest::fepois`) to handle county-year zeros without `log(Y + 1)` or `asinh`
distortions, and freezes all baseline exposures strictly over `2008` and `2010`
(preserving `AEWR-001`, where canonical `2009` statutory AEWR is strictly NULL;
see the [AEWR input contract](@/contracts/aewr-inputs.md)):

- **`E_p25` (primary, `3,108` counties)**: `2008/2010` county mean of
  $\ln(\text{AEWR}^{\text{PPI}}_{rt} / w^{\text{PPI}}_{p25,ct})$, standardized
  to mean `0` and SD `1` within each of the `17` AEWR regions. Within-region
  standardization algebraically cancels both the regional AEWR level and the
  national PPI deflator identically
  ($E_{p25} \equiv -\operatorname{std}_r(\overline{\ln w_{p25}})$, verified to
  `< 1e-12`).
- **`E_oews` (`3,066` counties)**: `2008/2010` county mean of
  $\ln(\text{AEWR}_{rt} / w^{\text{OEWS Big-Six}}_{ct})$, standardized within
  AEWR region.
- **`E_sf_log` (`3,108` counties)**: continuous all-real five-quantile
  agricultural shortfall comparing PPI-2012 `aewr_ppi` and
  `prevailing_ag_min_wage_ppi` against PPI-2012 ACS commuting-zone quantiles
  `wage_p10`--`wage_p90` with midpoint trapezoid weights
  `(0.175, 0.200, 0.250, 0.200, 0.175)`, averaged over `2008/2010` and
  standardized within AEWR region (matching the all-real unit convention of the
  [Mundlak–Chamberlain branch](@/designs/mundlak-chamberlain.md)).

{{ <grounding path="code/designs/bite/helpers.R" anchor="bite-estimation-contract" sha256="e6e7df32debbf2f37e2a5cf276a4d51f0da628ab6a1b633a22336631ef8d1f87" /> }}

Under the primary `1b-PPML` specification (`County + Region x Year FE` over
`2011--2025`), the coefficient on $E_c \times \ln(\text{AEWR}_{rt})$ identifies
the **relative differential elasticity** per `+1` region-specific SD of baseline
exposure $E_c$ between higher-bite and lower-bite counties within the same AEWR
region. It is **not an aggregate AEWR elasticity**: any common regional response
is absorbed by `Region x Year FE`, and the branch does not supply an
assumption-free average elasticity for the narrow welfare model.

The branch adds no new causal outcome to the canonical thirteen-outcome union
documented in [Causal outcomes and model inputs](@/contracts/causal-outcomes-and-model-inputs.md).

## Specifications and small-cluster inference

Every regression reports both cluster-robust CRV1 SEs and exact
delete-one-cluster jackknife (**CV3J**) SEs (`G = 17` AEWR regions for all AEWR
bite models; `G = 48` contiguous states for the standalone domestic-floor model)
with $t(G - 1)$ critical values, `MDE = 2.8 x CV3J SE`, and exact leave-one-cluster-out
(`LOO`) min/max ranges.

- **Primary `1b-PPML` (`E_p25 x ln(AEWR)`, `County + Region x Year FE`, `2011--2025`)**:
  - Certified hours (`N = 44,805`, `2,987` ever-positive H-2A counties):
    `+0.003` (CRV1 `0.129`, CV3J `[0.135]`, `p = 0.985`, MDE `0.379`, LOO
    `[-0.059, +0.091]`).
  - Certified positions (`N = 44,805`, `2,987` counties): `-0.010` (CRV1 `0.113`,
    CV3J `[0.119]`, `p = 0.931`, MDE `0.333`, LOO `[-0.059, +0.059]`).
  - Extensive margin (`1[Positions >= 5]`, LPM `feols`, `N = 46,620`, all
    `3,108` counties): `-0.024` (CRV1 `0.031`, CV3J `[0.032]`, `p = 0.454`, MDE
    `0.089`).
  - Implied duration `ln(Hours / Position)` ($\hat\theta_H - \hat\theta_N$,
    paired cluster jackknife): `+0.013` (CV3J `[0.052]`, `p = 0.803`, MDE
    `0.144`).
  - Precision across a `+2` SD baseline-exposure contrast (`+1` SD vs `-1` SD
    within region): the primary hours 95% CV3J interval (`t(16) = 2.120`) spans
    `[-0.57, +0.58]`, ruling out large differential collapses (`|2 theta| > 0.58`)
    while leaving modest differential elasticities statistically
    indistinguishable from zero.
- **Main alternatives (Hours / Positions, CV3J SE)**:
  - `State x Year FE` (`E_p25`): `-0.045 [0.158]` / `-0.047 [0.148]` (retaining
    `86.1%` unweighted / `86.3%` PPML-hours-weighted of `C + RY` variance).
  - `+ X_pre x Year FE` (`2,969` counties, interacting predetermined `2008--2010`
    `ln_pop_2010`, `farm_emp_share_0810`, and `ln_emp_farm_0810` with year
    indicators so `2023--2025` are preserved without discontinued BEA controls):
    `+0.029 [0.130]` / `-0.001 [0.110]`.
  - `E_oews` (`2,948` counties): `+0.041 [0.134]` / `+0.105 [0.162]`.
  - `E_sf_log` (`2,987` counties): `+0.008 [0.124]` / `-0.009 [0.122]`.
  - Transparent `1a-PPML` (`+ E_p25 x Year FE`): `-0.214 [0.752]` (MDE `2.106`)
    / `-0.073 [0.724]` (MDE `2.026`). Adding `E_p25 x Year FE` leaves only
    `2.61%` unweighted (`3.44%` PPML-hours-weighted) of the `C + RY`
    residualized regressor variance (`R^2(C + RY) = 0.9955` vs
    `R^2(C + RY + E x Y) = 0.9999`), explaining why `1a-PPML` is unpowered.

## Diagnostics, farm outcomes, and standalone domestic-floor margin

- **Event study (`2008--2025`, ref. `2011`)**: Joint `2008--2010` pre-trend
  Wald/F tests do not reject zero (`F_CV3J(3, 16) = 0.531, p = 0.667` for hours;
  `F_CV3J(3, 16) = 0.460, p = 0.714` for positions), with the explicit caveat
  that `2008--2010` weekly hours are pooled-mean imputed and `2009--2010` fall
  under the OES-to-FLS rule reversal.
- **Concentration and dependence**: Fisher-information effective clusters are
  `G_eff = 13.36` for hours (CA Region 17 share `12.1%`, Hessian identity
  relative error `1.35e-9`) and `12.25` for positions (CA share `13.1%`).
  Dropping the top `1%` H-2A counties (`32` counties) yields `+0.161 [0.125]`
  (hours) and `+0.100 [0.122]` (positions). State-year deviance-residual mean
  squares exceed county noise by `2.85x` (hours, `p = 1.43e-81`) and `2.29x`
  (positions, `p = 2.95e-48`).
- **Functional form and price decoupling**: Quartile SPT interactions
  (`Q2..Q4` vs `Q1`, hours) are non-monotonic and imprecise (`+0.227 [0.319]`,
  `-0.317 [0.433]`, `+0.094 [0.496]`). Decoupling the `WPU01` farm-products PPI
  (`E_p25 x ln(AEWR) + E_p25 x ln(PPI)`) yields `+0.037 [0.183]` on AEWR and
  `-0.048 [0.102]` on PPI for hours.
- **Addendum-B extensive-margin and FY2020 linkage caveats**: Multi-county
  fractional allocations inflate `any_h2a_gt0` (`+0.040* [0.020], p = 0.067` in
  `2011--2025` vs `+0.019 [0.060]` in `2011--2019`), whereas `any_h2a_ge5` is
  `-0.024 [0.032]` (`2011--2025`) and `-0.052 [0.051]` (`2011--2019`). Because
  Addendum-B disclosure expands worksite-business reporting at FY2020,
  `nbr_employers_balanced_start_year` is split strictly into `2011--2019`
  (`+0.346 [0.269]`) and `2020--2025` (`+0.083 [0.160]`).
- **Farm-economic outcomes (`E_p25 x ln(AEWR)`, `County + Region x Year FE`)**:
  real Fisher crop price (`2011--2024`, `-5.680 [4.280]`), farm hired-labor
  expense share (`2011--2022`, `+0.0043 [0.0032]`), and BEA farm employment
  (`2011--2022`, PPML `+0.0014 [0.0146]`, MDE `0.041`; OLS `+7.538 [13.995]`).
- **Standalone domestic-floor margin (`2011--2025`, all `48` contiguous states)**:
  Unlocked by the unified A01 FRED minimum-wage series, FE-PPML regressions of
  certified hours (`+0.376`, CRV1 `0.228`, CV3J `[0.374]`, `p = 0.319`, MDE
  `1.046`) and certified positions (`+0.313`, CRV1 `0.142 [p = 0.027]`, CV3J
  `[0.212]`, `p = 0.147`, MDE `0.594`) on `ln(prevailing_ag_min_wage)` under
  `County + Region x Year FE` use state CRV1 and state CV3J jackknife SEs (`46`
  multi-state-region states contribute within-region variation; CA and FL are
  absorbed). This is reported strictly as a standalone domestic-floor reduced
  form and is never translated into an AEWR elasticity.

## Retained artifacts and validation

Stage `03_validate_bite.R` independently verifies string geographic keys, exact
`55,944`-row balance, `AEWR-001` (`2009 aewr` NULL), the algebraic PPI-invariance
identity (`< 1e-12`), all-real `E_sf_log` unit parity (`< 1e-12`), finite CV3J
SEs and MDEs, `2025` preservation under `+ X_pre x Year FE`, 48-state standalone
domestic-floor coverage, an independent `fixest::fepois` spot check, and
SHA-256 parity across `outputs/reports/bite_provenance.json` and all retained
outputs (`data/processed/bite_county_year_panel.parquet`,
`outputs/tables/bite/bite_{main_estimates,diagnostics,event_study,loo_regions}.csv`,
`outputs/tables/table_bite_{main,diagnostics}.tex`,
`outputs/figures/fig_bite_event_study.{pdf,png}`, and
`outputs/reports/bite_design.md`). Start-year `2025` totals remain provisional
pending FY2027 decisions.

## Power-rescue diagnostic archive

An isolated two-phase, outcome-blind power-rescue investigation
(`explorations/bite_power_rescue/`, summarized in
`markdowns/bite_rescue_power_report.md`) audited 59 candidate extensions across
six families (`R1`--`R6`: minimum-wage floor truncation and sub-state tiers,
direct-grower restrictions with CDL/NOAA covariates and ANCOVA/SDID/Bai IFE,
balanced worksite-business cohorts and within-multi-state-filer reallocation,
duration-amortization and farm labor-share bite, 2025 local-wage crossover bite,
and stacked 2009--2010 plus 2025 reversals) using cached public data only. Zero
Tier 1 (`MDE <= 0.40`) or 2025-rule upgrades cleared size-valid Phase 1 power
gates (`55/59` killed outcome-blind), `R4a` duration bite failed Phase 2 referee
audit on mechanical duration mean reversion, and `R2a` (`2020--2025`
direct-grower `E_p25` bite) remained a diagnostics-only null without a wage
first stage. Neither production branch (`code/designs/policy_2025`,
`code/designs/bite`) was replaced.
