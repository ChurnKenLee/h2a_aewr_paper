# Panel-IV instructions

- This branch consumes the shared panel directly; it cannot use DiD treatment
  classifications or the DiD panel.
- Preserve target/donor construction, the wage-only and
  wage-plus-seasonal/composition instruments, prior-period timing, soft
  entropy calibration with `rho = 0.10`, and common samples.
- The calibration unit is county, with one normalized distribution per AEWR
  region and source year. Do not recover or reuse OEWS-area weight partitions.
- The preferred moments are three independent FLS/QCEW seasonal contrasts and
  four undivided FLS field/livestock composition residuals. Do not infer,
  divide by, or clip field-worker shares from rounded wage rates.
- QCEW 111/112 remains primary for county employment paths, quarterly
  seasonality, and field/livestock composition. Donor wage levels use only the
  county-mapped OEWS-area Big-Six hourly-wage proxy; do not describe that proxy
  as a direct county farm-wage observation or add a QCEW/BEA wage fallback.
- Preserve target-area overlap exclusion, PPI deflation, and source year `t-1`.
- The static-propensity interaction is a control in both stages, never an
  excluded instrument.
- Column 4 remains the preferred controlled
  wage-plus-seasonal/composition specification.
  Do not relabel preferences based on favorable estimates or first stages.
- First-stage strength, cluster counts, target/donor diagnostics, and
  identical-sample comparisons travel with estimate changes.
- Preserve restartable caches, atomic Parquet writes, deterministic ordering,
  supported-year guards, and memory-aware behavior in Python recovery steps.
