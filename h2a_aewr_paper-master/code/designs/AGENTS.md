# Empirical-design instructions

- Each subdirectory owns a distinct estimand, treatment construction, sample, fixed effects, inference rule, diagnostics, and outputs. Shared outcomes may be consumed; design objects may not be shared implicitly.
- Before changing a design, write down the estimand and enumerate treatment timing, sample years, normalization, controls, fixed effects, clusters/reference law, weights, support restrictions, and output tables affected.
- Treat numerical equivalence as a testable claim. Use identical-sample comparisons and tolerances tied to the estimand, not visual similarity of tables.
- Preserve diagnostics that challenge the preferred interpretation. Placebo failures, weak first stages, rank ceilings, support violations, and out-of-range linear predictions are findings, not cleanup targets.
- Update the assumption registry and design-specific agent page whenever a substantive design choice changes.
