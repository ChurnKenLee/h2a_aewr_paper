# Empirical-design instructions

- Each design owns its estimand, treatment construction, sample, fixed effects,
  inference, diagnostics, and outputs. Do not share design objects implicitly.
- Before changing a design, enumerate treatment timing, sample years,
  normalization, controls, fixed effects, clusters or reference law, weights,
  support restrictions, and affected outputs.
- Treat numerical equivalence as testable. Compare identical samples with a
  tolerance tied to the estimand rather than visually comparing tables.
- Preserve diagnostics that challenge the preferred interpretation. Placebo
  failures, weak first stages, rank ceilings, support violations, and
  out-of-range predictions are findings, not cleanup targets.
- Update the canonical design page whenever a substantive choice changes.
- Keep `content/contracts/causal-outcomes-and-model-inputs.md` aligned with
  executable outcome registries, transformations, denominators, sample rules,
  and required estimation fields.
