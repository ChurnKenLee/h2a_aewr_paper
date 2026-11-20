# Panel-IV instructions

- Retain the original moving-wage design. The fixed-wage revision,
  separation/innovation exercises and simulated calibration draws were removed
  by explicit user instruction. Do not restore those branches or their outputs.
- Stage 10 is a user-requested diagnostic: the Kitagawa decomposition of the
  instrument into calibrated-weight, donor-wage and interaction components and
  the share of identifying variance each carries. It uses the recovered
  weights as-is (their response to wage features is intended), never
  recalibrates with frozen wages, never replaces the instrument or a column and
  never selects a model. Keep its 2008--2010 base wage, balanced-donor rule and
  source-year-mean reference weights unless a revision is declared. The former
  component 2SLS, Anderson--Rubin sets and Hansen J were removed to keep it
  simple; do not restore them unasked.
- Stage 11 is a user-requested diagnostic of donor-wage granularity. It
  multiplies the OEWS wage by an annual county QWI NAICS 111/112 factor in
  memory, recalibrates the weights on that wage with stage 04's moments and
  solver, and compares the result with the design. By user instruction it
  reports only that recalibrated annual-factor variant beside the design; the
  fixed-factor, design-weight and QWI-outright variants were dropped. It
  rebuilds the instrument with stage 05's rules, repeats stage 10's
  decomposition and reports preferred-column first stages for the instrument
  and its weight and wage components. The OEWS proxy stays the only wage of
  the design: the stage writes no calibration or instrument artifact,
  estimates no outcome equation, and no result selects a wage source. It must
  reproduce stages 04, 05 and 10 with the OEWS wage; keep those guards when
  any of them changes.
- By explicit user instruction (2026-11-09) the design has one instrument,
  `z_wage_seasonal_composition_real`, estimated in two columns: column 1
  without controls and column 2 with controls (preferred). The wage-only
  calibration, its `z_wage_only_real` field and estimation columns, and the
  `continuing` control set were removed; do not restore them or their
  `_continuing_controls` outputs without a declared revision. Stage 04
  recovers only the wage-plus-seasonal/composition distribution, and the
  validator rejects any other calibration or instrument specification.
- Keep all branch implementation and tests in R, using plain functions and
  the existing configured packages. JSON registries and shell runners remain.
  Do not add Python runtime bridges or custom NumPy/C compatibility adapters.
- Read `estimation_spec.json` through `specification.R`; it is the sole
  outcome/model registry. Preserve the twelve outcomes and reviewed v2
  worksite-business linkage, controls, equal county-year weights, common
  samples, county/year FE and target-cluster CRV1 inference. Controlled
  column 2 is preferred.
- Estimation policy years run 2011--2023 (source years 2010--2022): the
  committed BEA controls are the only control set, and CAEMP25N ends in 2022.
  Instruments and the county-year panel are still constructed through policy
  year 2024 (`DISSIMILARITY_IV_POLICY_END_YEAR`); the estimation window is
  `DISSIMILARITY_IV_ESTIMATION_END_YEAR` and `policy_years` in the registry.
- Consume the shared panel directly. Do not use DiD treatments or its panel.
- Preserve five target subregions per AEWR region, the two selected donor
  clusters, target-primary-OEWS-area overlap exclusion, source year t-1,
  source-year PPI deflation and original observed-current-wage donor support.
- County-mapped OEWS-area Big-Six hourly wages are the only donor wage proxy;
  they are not direct county farm wages. No QCEW/BEA wage fallback is allowed.
- QCEW 111/112 remains primary for county employment, seasonality and
  composition. Preserve three independent seasonal contrasts and four
  undivided composition residuals; never infer or clip shares from wage gaps.
- `frame.R` preserves the implemented benchmark fills, forward growth, unit
  annual raking, zero two-sided drift and historical last-record farm floor.
  The source-fingerprinted Census selection records the former Polars row
  choices; review source drift rather than silently regenerating that input.
  Historical metadata labels are not evidence of implemented state raking.
  Correcting these limitations is a separate empirical revision.
- `calibration.R` uses deterministic entropy centers with rho=.10 and the
  existing convergence tolerance and 100-iteration guard. Preserve zero-prior
  support, inactive moments and the documented April-2011 survey gap.
- Retain normalization, support, feature/moment-fit and concentration checks.
  Recovered weights are constructed pseudo-distributions, not observed
  sampling weights or established exogenous survey errors.
- Keep geographic keys as strings through shared normalizers. Preserve
  atomic Parquet writes, deterministic ordering and restartable stage04
  year/region subsets; reject stale schemas before writing a partial rebuild.
- The static-propensity interaction is a control in both stages, never an
  excluded instrument. No result or first-stage statistic selects a model.
- Validate changed behavior with focused R tests and the smallest affected
  branch run. Keep the grounded design page and assumptions aligned.
