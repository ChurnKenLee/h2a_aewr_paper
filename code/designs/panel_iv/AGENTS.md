# Panel-IV instructions

- This branch consumes the shared panel directly; it cannot use DiD treatment
  classifications or the DiD panel.
- Preserve target/donor construction, wage-only and wage-plus-seasonal
  instruments, prior-period timing, entropy calibration, and common samples.
- The static-propensity interaction is a control in both stages, never an
  excluded instrument.
- Column 4 remains the preferred controlled wage-plus-seasonal specification.
  Do not relabel preferences based on favorable estimates or first stages.
- First-stage strength, cluster counts, target/donor diagnostics, and
  identical-sample comparisons travel with estimate changes.
- Preserve restartable caches, atomic Parquet writes, deterministic ordering,
  supported-year guards, and memory-aware behavior in Python recovery steps.
