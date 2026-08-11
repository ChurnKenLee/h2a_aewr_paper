+++
title = "Assumption registry"
description = "Curated implementation assumptions tied to executable source checks."
weight = 4
+++

> [!NOTE]
> Generated file. Change repository sources or `scripts/agent_grounding.py`, then regenerate.

Every record below passed its checks when this page was generated. A passing check shows that code and the declared statement still agree syntactically; it does not prove the research assumption.

| ID | Status | Risk | Owner | Statement | Source |
|---|---|---|---|---|---|
| `AUTH-001` | active | critical | repository | Executable code, runners, locks, and checked contracts outrank prose when sources disagree. | `AGENTS.md` |
| `GROUND-001` | active | critical | agent grounding | Authoritative grounding code fences are source-derived, selector-addressed, syntax-checked, and blocked when their reviewed excerpt digest changes. | `scripts/agent_grounding.py` |
| `GROUND-002` | active | high | agent grounding | Literal code fences on agent-grounding surfaces are rejected unless they have a language and an explicit non-authoritative classification. | `scripts/agent_grounding.py` |
| `ROOT-001` | active | high | shared pipeline | The .here marker and code/paths.R define repository-root paths for supported R entry points. | `code/paths.R` |
| `GEO-001` | active | critical | shared pipeline | Persistent geographic identifiers are strings and county_fips uses the 2010 county vintage. | `documentation/geographic_code_contract.md` |
| `GEO-002` | active | critical | shared pipeline | Oglala Lakota County code 46102 is mapped to 2010-vintage Shannon County code 46113. | `documentation/geographic_code_contract.md` |
| `PRED-001` | active | critical | B01 prediction | The selected H-2A prediction trains through 2011 and uses model_spec climate_norm_static_v1. | `code/paths.R` |
| `PANEL-001` | active | critical | C02 shared panel | The shared county-year panel contains reusable measures but no design treatment, post indicator, fixed-effect encoding, target cluster, or instrument. | `code/c02_build/README.md` |
| `DID-001` | active | critical | DiD design | The baseline DiD post period begins after 2011. | `code/designs/did/01_build_did_panel.R` |
| `DID-002` | active | critical | DiD design | DiD models use county and year fixed effects and cluster by commuting-zone × AEWR-region interaction. | `code/designs/did/helpers.R` |
| `PIV-001` | active | critical | panel-IV design | The panel-IV policy window is 2011-2022, with five subregions and two primary donor subregions. | `code/designs/panel_iv/design.R` |
| `PIV-002` | active | critical | panel-IV design | The preferred instrument adds quarterly worker-share moments to the wage target and uses soft entropy rho=0.10. | `code/designs/panel_iv/design.R` |
| `PIV-003` | active | high | panel-IV design | The static H-2A propensity differential trend is a control, not an excluded instrument. | `code/designs/panel_iv/design.R` |
| `MC-001` | active | critical | Mundlak-Chamberlain design | The supported design version is 3.0.0 with 2008-2010 baselines, 2011-2022 treatment history, and 2013-2022 analysis years. | `code/designs/mundlak_chamberlain/design.R` |
| `MC-002` | active | critical | Mundlak-Chamberlain design | Finite-design CCV uses 17 balanced cyclic AEWR-path assignments and therefore 16 reference degrees of freedom. | `code/designs/mundlak_chamberlain/design.R` |
| `MC-003` | active | high | Mundlak-Chamberlain design | The default version-3 queue is compact and resource-guarded; exhaustive registry execution is opt-in. | `code/designs/mundlak_chamberlain/design.R` |
| `DOC-001` | historical | critical | documentation | The 2023 H2A Estimation Strategy Word file proposes a border-enforcement shift-share design and is not a current supported runner. | `documentation/H2A Estimation Strategy.docx` |
| `DOC-002` | historical | critical | documentation | The 2024 Price Index Construction Word file proposes Nielsen/FAF gravity recovery; the supported derived pipeline uses the NASS/CDL Fisher implementation. | `documentation/Price Index Construction.docx` |
| `DATA-001` | active | high | repository | Raw and processed data are ignored local state; repository static checks do not prove empirical reproducibility. | `.gitignore` |

## Review triggers

- **AUTH-001** — Any source-of-truth or documentation workflow changes.
- **GROUND-001** — Snippet selection, rendering, validation, acceptance, or generated documentation changes.
- **GROUND-002** — Grounding Markdown scope or literal-fence policy changes.
- **ROOT-001** — Path discovery or repository layout changes.
- **GEO-001** — Any producer, crosswalk, join key, or geographic vintage changes.
- **GEO-002** — County-vintage normalization changes.
- **PRED-001** — Prediction training, scoring, cutoff, or shared-panel join changes.
- **PANEL-001** — The shared-panel schema or any design input changes.
- **DID-001** — Treatment timing or the panel calendar changes.
- **DID-002** — Formula or inference code changes.
- **PIV-001** — Clustering, donor selection, or the policy calendar changes.
- **PIV-002** — Moment construction, calibration, or preferred-specification labels change.
- **PIV-003** — Controls, first-stage formulas, or prediction use changes.
- **MC-001** — Calendar, lags, design version, or registry construction changes.
- **MC-002** — Reference law, assignment units, covariance, or critical values change.
- **MC-003** — Execution stages, resource guards, or registry size changes.
- **DOC-001** — A border-enforcement design becomes executable or the file is reclassified.
- **DOC-002** — The supported price-index producer changes.
- **DATA-001** — Data distribution or fixture policy changes.
