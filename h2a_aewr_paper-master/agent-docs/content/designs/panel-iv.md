+++
title = "Panel IV"
description = "Dissimilarity subregions, FLS/OEWS recovery, entropy calibration, 2SLS, and diagnostics."
weight = 2
+++

The panel-IV section of [source-linked code grounding](@/generated/code-grounding.md)
grounds the frozen constants, occupation frame, control/trend contract,
identifiers, weighting, estimation skeleton, sample, and coefficient guard.

Canonical sources: `code/designs/panel_iv/design.R`, the branch README, and
the eight numbered runner stages.

## Current contract

- Policy window: 2011-2022; feature window: 2008-2011.
- Five crop/climate/soil subregions per AEWR region; two primary donors.
- Wage-only and wage-plus-quarterly-worker-share instruments share the Census hired-worker frame prior.
- Preferred calibration uses soft entropy with `rho = 0.10`.
- Preferred publication column is the controlled wage-plus-seasonal specification.
- County/year fixed effects; inference clusters by `aewr_iv_cluster_id`.
- The static H-2A propensity × centered-year term is a control in both stages, never an excluded instrument.

## Change hazards

Target/donor recovery is a design object, not generic cleaning. Preserve source
timing, cache/restart behavior, atomic writes, and diagnostic parity. Compare
specifications on identical samples. First-stage strength does not by itself
validate exclusion or justify relabeling a preferred specification.
