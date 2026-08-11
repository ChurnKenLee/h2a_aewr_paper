# Shared R helpers

This directory contains only design-neutral helpers sourced by C-side scripts.

| File | Responsibility |
| --- | --- |
| `geography.R` | Normalize and combine persistent geographic identifiers |
| `bea_county_crosswalk.R` | Harmonize BEA county definitions |

Design-specific constants and numerical methods live with their owner under
`code/designs/`.

Canonical representations, normalization rules, and cross-language grounding
are documented in the
[geographic code contract](../../content/contracts/geographic-codes.md).
