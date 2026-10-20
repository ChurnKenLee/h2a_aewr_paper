# Panel-IV instructions

- Retain the original moving-wage four-column design. The fixed-wage revision,
  separation/innovation exercises and simulated calibration draws were removed
  by explicit user instruction. Do not restore those branches or their outputs.
- Exception, re-authorized by the user: stages 10--11 run the Kitagawa
  decomposition of the retained instruments into calibrated-weight,
  donor-wage and interaction components. It is diagnostic only. It uses the
  recovered weights as-is (their response to wage features is intended), never
  recalibrates with frozen wages, never replaces an instrument or column and
  never selects a model. Keep its 2008--2010 base wage, balanced-donor rule and
  source-year-mean reference weights unless a revision is declared.
- Keep all branch implementation and tests in R, using plain functions and
  the existing configured packages. JSON registries and shell runners remain.
  Do not add Python runtime bridges or custom NumPy/C compatibility adapters.
- Read `estimation_spec.json` through `specification.R`; it is the sole
  outcome/model registry. Preserve the twelve outcomes and reviewed v2
  worksite-business linkage, controls, equal county-year weights, common
  samples, county/year FE and target-cluster CRV1 inference. Controlled
  seasonal/composition column 4 remains preferred.
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
