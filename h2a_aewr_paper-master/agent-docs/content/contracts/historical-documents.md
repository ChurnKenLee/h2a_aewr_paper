+++
title = "Historical and proposed documents"
description = "A quarantine map for dated design notes that no longer describe the supported implementation."
weight = 3
+++

## 2023 H2A Estimation Strategy Word note

Status: **historical/proposed**.

The note proposes a county-panel shift-share IV based on border-enforcement
exposure, undocumented-flow estimates, and lagged/initial undocumented shares.
No border-enforcement runner is present in `scripts/run_all.sh`. Current
supported empirical branches are DiD, panel IV based on FLS/OEWS geography,
and Mundlak–Chamberlain dose response.

## 2024 Price Index Construction Word note

Status: **historical/proposed**.

The note proposes recovering farm-gate prices from Nielsen retail scanner data,
Freight Analysis Framework commodity flows, distance/gravity models, and
transport costs. The supported B01 price/output producer is currently
`code/b01_derived/02_price_index_nass_synthetic_cdl.py`, a NASS/CDL chained-Fisher construction.

## Safe use

Historical notes may explain motivation or abandoned alternatives. They must
not supply current paths, inputs, estimands, or implementation details unless a
current executable source independently confirms the claim.
