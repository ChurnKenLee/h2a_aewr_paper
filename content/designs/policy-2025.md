+++
title = "2025 H-2A policy shift"
description = "Separate design for offered wages and covered crop earnings, preserving the inspected specifications and partial window."
[extra]
scopes = ["code/designs/policy_2025", "scripts/run_policy_2025.sh", "outputs/reports/policy_2025.md"]
+++

`code/designs/policy_2025` owns the current wage-offer and covered-earnings
analyses of the October 2, 2025 shift. The user requested this separate design
on November 16, 2026. This is an organizational migration after inspecting
outcomes, not a new preregistration or a change of empirical specification.
The [earlier chapter exploration](@/designs/chapter-directions.md) remains a
historical record; its 2013 administrative-interruption analysis is excluded.
The historical wage-reform and statewide-border power designs remain separate.

The registry and its historical declarations fix the estimands, source vintage,
weights and bounded queue. The branch README owns commands, stage order and
artifact locations. Its standalone runner consumes local source-owned inputs,
uses isolated temporary work, checks equivalence before publication and records
input/code/output hashes. It is outside the complete pipeline, manuscript and
welfare runners. Migration alone does not promote a causal claim.

{{ <grounding path="scripts/run_policy_2025.sh" anchor="policy-2025-runner" sha256="438ccef5680a68ea107364f606a6093e3783d166aafbb909311f11c82597263d" /> }}

## Offered-wage estimand

The primary outcome is the nominal hourly offer change between FY2025 and
FY2026 for continuing R7 filer–state–occupation cells. Fiscal decisions use
October 1 through June 30 in both years. The dose is the U.S.-worker skill-I
statewide reference minus the FY2025 regional rate; a cut is negative.
Former AEWR-region effects and baseline certified-hours weights are retained.
Counties have positive fixed-2011 farm employment and belong to multi-state
former regions; Alaska, Hawaii and DC are excluded. Source hours retain the
historical pooled imputation and allocation convention.

Equal weights, R5/R8/disclosed FEIN, fixed baseline occupation composition,
within-filer across-state comparisons, 2024/2025 placebos, selection,
distribution and applicability audits remain in the complete ledger. Three
baseline heterogeneity interactions retain Holm-adjusted p-values and
Bonferroni simultaneous intervals. No specification is chosen by significance.

R7 identifies filers, not the source-owned worksite businesses. Shared name
normalizers are reused, while the design owns these unchanged filer rules.
Continuation and common occupations condition on observed survival; neither
defines a population effect without further assumptions. Exact job-level
legal floors stay unknown because receipt is not job-order submission and
skill, worker category, prevailing wage and itineraries remain unresolved.
The four-schedule/minimum-wage envelope is explicitly a reference diagnostic.

## Covered-earnings estimand

The current primary is the user-authorized November 14 interim window:
the mean of the 2025Q4 and 2026Q1 year-over-year log changes in private crop
production average weekly earnings. Each county must have both current/prior
pairs, positive disclosed earnings and employment of at least ten in all four
observations. States require at least two eligible counties. Missing periods
stop the requested window; missing county observations exclude the county
from that window and are never imputed.

{{ <grounding path="code/designs/policy_2025/interim_helpers.R" anchor="policy-2025-interim-window" sha256="27e55643bb8c0dbfd9870c66c677fe945a396b263cda54467dfb249830cd921d" /> }}

Frozen intensity is mean 2022–2024 H-2A certified positions per fixed-2011 farm
employment, winsorized at its original 99th percentile. The exact original
county exposure/dose table is retained with SHA-256
`6bd476d1bf0c4cb8d98e3072d1b7b1d6375176d18dffc62a71d6bb6e0e25eda3`.
The regressor is intensity times the statutory dollar cut, with intensity and
state effects included and equal county weights. There is no added trend,
instrument, normalization, seed, new resource guard or sample restriction.

Employment weights, H-2A cash dose, commuting-zone exposure and omission of
the intensity control remain sensitivities. The design also retains separate
quarters, the same two-quarter sample in each quarter, the full prior-window
lead, matched Q4/Q1 placebos for 2016–2024, the earlier three-quarter pre-policy
diagnostics, frozen UI splits, animal and all-private comparisons. QWI is
secondary and unavailable unless the same declared periods are present.

The earlier full target is 2025Q4–2026Q2 with all six prior/current observations.
Its implementation and publication guard are preserved, and its historical
acquisition command remains separately documented. It is not invoked by the
current runner. The interim and any later full-window update are multiple
looks at already inspected outcomes, not independent confirmations.

{{ <grounding path="code/designs/policy_2025/helpers.R" anchor="policy-2025-complete-window" sha256="51503cc63e3c408ca15f7621b3026f35d2590fce8a7127d37bf993aca976e50c" /> }}

## Inference and interpretation

All migrated fits retain state deletion jackknife inference with t(G−1)
intervals and CRV1 as a secondary comparison. Any unidentified deletion leaves
the jackknife interval unavailable; failed replicates are not silently dropped.
Every failed fit remains in the ledger. The fixed-effects small-sample
conventions and one-thread setting are explicit in this branch's helpers.

Causal interpretation needs exposure-related untreated changes to be unrelated
to the dose after controls and fixed effects. Differential state shocks and
contemporary policy changes remain concerns. The initial wage extensions were
developed after related FY2026 findings were seen. Quiet pre-policy diagnostics
cannot establish exogeneity. Weekly earnings mix wage rates, hours and workforce
composition; incomplete UI coverage cannot establish domestic-worker effects.
No welfare elasticity or general labor-market spillover follows automatically.

## Migration validation and ownership

The new runner requires the original source hashes, freezes exposure and UI
classification, and compares all 347 wage, 135 pre-policy and 46 interim
coefficient rows to the inspected reference ledgers at absolute tolerance
1e-10. Samples, formulas, weights, failed statuses, p-values, standard errors
and confidence intervals are included. Independent base-R WLS/OLS,
allocation conservation, CDF ordering, complete periods, suppression and
canonical string-key checks are retained.

The branch writes filer exchange artifacts to `data/intermediate/policy_2025`,
analysis panels to `data/processed/policy_2025`, and estimates/audits and reports
to `outputs`. The publisher verifies copied bytes and retains a complete
manifest. A source refresh or empirical revision requires a dated amendment
and preserved previous results; it must not silently replace the migration
comparison fixture or change an upstream schema in this consumer.
