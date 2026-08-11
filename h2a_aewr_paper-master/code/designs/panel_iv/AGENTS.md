# Panel-IV instructions

- This branch consumes the shared panel directly; it must not use DiD treatment classifications or the DiD panel.
- Preserve the fixed target/donor construction, the wage-only and wage-plus-seasonal instruments, prior-period source timing, entropy-calibration contract, and common-sample comparisons.
- The controlled specifications include the documented lagged controls and static-propensity differential trend. The propensity interaction is a control in both stages, never an excluded instrument.
- Column 4 is the preferred controlled wage-plus-seasonal specification. Do not relabel preferences because a different column has a larger first-stage statistic or more favorable estimate.
- First-stage strength, cluster counts, donor/target diagnostics, and identical-sample static-trend comparisons must travel with estimate changes.
- `04_recover_fls_geography.py` is large, numerically sensitive, and restartable. Preserve atomic Parquet writes, supported-year guards, cache reuse, deterministic ordering, and memory-aware implementation.
