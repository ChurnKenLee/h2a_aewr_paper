+++
title = "Research integrity and interpretation"
description = "Changes that require substantive review and diagnostics that must not be optimized away."
weight = 2
+++

## Substantive change gate

Treat changes to any of the following as research-design changes, not refactors:

- treatment level, timing, lags, polynomial basis, or reference period;
- sample years, eligibility, missing-data handling, or common-sample rules;
- outcome definition, unit, normalization, denominator, or aggregation;
- controls, fixed effects, trends, clusters, instruments, weights, or covariance;
- support/extrapolation rules, model-selection rules, or critical values.

Write down the old and new estimands and update the assumption registry before
claiming equivalence.

## Diagnostics are results

Weak first stages, rank constraints, collinearity, support violations, placebo
rejections, out-of-range linear predictions, unstable functional forms, and
model warnings are not nuisances to suppress. A software-validation pass
establishes implementation consistency; it does not certify identification.

## Manuscript reconciliation

Every quantitative claim should resolve to an owning output and producing
script. The draft is downstream of code and artifacts. Never edit a table value
or causal caveat by hand to reconcile prose.
