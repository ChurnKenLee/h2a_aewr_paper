# Mundlak–Chamberlain instructions

- Version 3 is the supported runner path. The older `02_estimate_models.R`, `03_postestimation.R`, `04_diagnostics.R`, and `05_validate.R` files are retained benchmark/compatibility code unless the task explicitly targets them.
- Preserve the declared registry, finite-design region-path reassignment, common full-rank coefficient basis, named causal dictionary, exact analytic gradients, and resource guards.
- The exhaustive registry is not the default execution queue. Do not trigger all declared specifications without an explicit resource decision.
- The finite-design CCV has assignment-level rank limitations; county replication does not remove them. Never substitute ordinary clustered covariance while retaining the CCV label.
- Focal dose coordinates must remain separated from nuisance history coordinates to avoid exact collinearity. Do not allow automatic collinearity dropping to decide the causal basis.
- Keep support tables, lead placebos, out-of-range linear-probability diagnostics, model warnings, and selected-primary logic visible even when they undermine a preferred causal narrative.
- Validation must cover registry/rank/row guards, common bases across reassignment states, variance adjustment, gradients, scale conversions, and checkpoint compatibility.
