+++
title = "Difference in differences"
description = "Baseline continuous treatment, event study, samples, fixed effects, and clustered inference."
weight = 1
+++

The DiD section of [source-linked code grounding](@/generated/code-grounding.md)
shows the current treatment classifier, sample restriction, main formula, and
event-study formula directly from their R sources.

Canonical sources: `code/designs/did/README.md`, `01_build_did_panel.R`, and
`helpers.R`.

## Current contract

- Consumes `data/processed/county_year_panel.parquet`.
- Defines `post = year > 2011`.
- Uses the lagged AEWR minus commuting-zone 25th-percentile wage gap.
- Uses county and year fixed effects.
- Clusters by commuting-zone × AEWR-region interaction.
- Retains full/no-border × uncontrolled/controlled comparisons.
- Event study uses 2011 as the reference and keeps its explicit slope handling.

## Change hazards

Do not turn the continuous treatment into an event without redefining the
estimand. Do not move treatment or post indicators into the shared panel.
Preserve common-sample comparability, event-time indexing, and retained
pre-trend evidence when formulas change.
