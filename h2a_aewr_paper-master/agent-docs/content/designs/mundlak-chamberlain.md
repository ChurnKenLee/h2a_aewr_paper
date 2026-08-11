+++
title = "Mundlak–Chamberlain"
description = "Version-3 specification program, dose-response basis, finite-design CCV, support, and resource guards."
weight = 3
+++

The Mundlak-Chamberlain section of [source-linked code grounding](@/generated/code-grounding.md)
grounds the version/calendar, registry and execution queues, rank and memory
budgets, balanced reference states, and finite-design CCV covariance.

Canonical sources: `code/designs/mundlak_chamberlain/design.R`,
`specification_program.R`, the branch README, and the version-3 runner stages.

## Current contract

- Baseline histories: 2008-2010; treatment history: 2011-2022; analysis: 2013-2022.
- Program version 3.0.0; compact default queue; exhaustive registry is opt-in.
- Current, one-year, and two-year lag dose coordinates, with an identified quadratic primary basis.
- Eight outcomes and a declared model ladder/registry.
- Finite-design covariance reassigns the 17 complete AEWR paths under a balanced cyclic reference law, giving 16 reference degrees of freedom.
- Named analytic gradients and common full-rank bases are validated.
- Dense-matrix and estimated-peak memory guards prevent unsafe fits.

## Interpretation boundary

The design conditions on a rich history/projection; it does not make arbitrary
time-varying confounding disappear. County replication does not increase the
number of independent AEWR policy paths. Support failures, lead-placebo
rejections, and out-of-range linear effects remain visible even when software
validation passes.
