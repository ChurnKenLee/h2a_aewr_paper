+++
title = "Data and geography"
description = "Persistent identifiers, source ownership, raw-data safety, credentials, and external side effects."
weight = 1
+++

The geography/data section of [source-linked code grounding](@/generated/code-grounding.md)
contains the paired R/Python normalizers and assertions, price-fallback joins,
and shared-panel output guards. Those fences are authoritative projections of
the current source rather than copied examples.

## Geography

The active source is `documentation/geographic_code_contract.md`.

- Persistent geographic fields are strings.
- `state_fips`, `county_code`, and `county_fips` preserve their documented padding.
- County geography is harmonized to the 2010 vintage.
- The current Oglala Lakota code `46102` maps to 2010-vintage Shannon County `46113` where the contract specifies.
- R producers use `code/c00_shared/geography.R`; Python producers use `h2a.geography`.

## Data tiers

- `data/raw`: downloaded sources and hand-maintained crosswalks. Never destructively refresh by default.
- `data/intermediate`: exchange artifacts with one owner; fix the producer.
- `data/processed`: analysis-ready panels with declared keys and schemas.
- `outputs`: retained manuscript products; regenerate from their owner.

The archive does not contain `data/`. A dry run or syntax pass therefore says
nothing about empirical completion.

## External operations

Before a source run, identify the endpoint/provider, credential variable,
request count, monetary/quota risk, cache, resume behavior, output target, and
overwrite policy. The structured source inventory is
`documentation/raw_data_sources.yaml`.
