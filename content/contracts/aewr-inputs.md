+++
title = "AEWR input history and applicability"
description = "Source-correct annual references, dated wage lookup tables and explicit coverage gaps."
[extra]
scopes = ["code/a01_sources/01_aewr_extract_tables.py", "src/h2a/aewr.py", "code/c01_clean", "code/designs"]
+++

The source producer preserves annual FLS reference rates for 49 states from
1990 through 2025. Rates through 2008 come from CRS Table 1; later rates come
from explicitly selected Federal Register state tables. The USDA workbook is
an independent audit and never selects the output sample. The Hawaii 2003
correction is sourced to document 03-6559.

{{ <grounding path="code/a01_sources/01_aewr_extract_tables.py" anchor="aewr-source-contract" sha256="036f1457a6c43aa7f815391898b7ab7b07ca67db43e9c44ed4b8276fd79aa24b" /> }}

The 2009 OES regime did not imply one rate per state. Table 1 of the
[2010 final rule](https://www.govinfo.gov/content/pkg/FR-2010-02-12/pdf/2010-2731.pdf)
reports application wage means. Canonical 2009 AEWR is therefore null, and
those means are retained only under their nonstatutory name. The scheduled
June 2009 suspension was enjoined and never implemented, as documented by the
[court chronology](https://www.govinfo.gov/content/pkg/USCOURTS-ncmd-1_09-cv-00411/pdf/USCOURTS-ncmd-1_09-cv-00411-3.pdf).
Exact OES area/occupation/skill determinations and transition cohorts are still
needed to reconstruct legal job exposure in that period.

The tracked notice registry records publication dates separately from effective
dates. `aewr_rate_schedule.parquet` uses half-open intervals, beginning in 1995,
and preserves the 2009--early-2010 gap. Early numeric rates come from CRS;
2000 onward tables are parsed from cached notices. These are FLS references,
not universal job-level obligations. December 2024 has a separate possible
December 30 effective date for injunction-covered states/entities. Do not infer
entity eligibility from a state code. Pre-1995 effective dates remain unknown.

The cached 2025--2027 occupational workbooks retain state, SOC/category, skill,
U.S./H-2A worker category, hourly rate, source text, source sheet and effective
interval. The [FLAG archive](https://flag.dol.gov/wage-data/adverse-effect-wage-rates)
identifies the two release intervals; the current release has no assumed future
end date. The preceding 2023--2025 other-occupation workbooks were unavailable
locally and network acquisition failed. Missing lookup cells are not zero.

Actual obligations additionally require job duties, job-order cohort, range
status, applicable prevailing and collective-bargaining wages, federal/state
minimum-wage coverage, promised wages and injunction membership. Annual state
minimum-wage references do not establish these job-specific conditions.
`aewr_policy_events.parquet` is a sourced ledger with concurrent changes and an
explicit absence of any exogeneity assumption. Annual FLS/OEWS changes can
reflect the same labor-demand shocks as H-2A utilization.

All inputs carry SHA-256 provenance in `aewr_source_manifest.json`. Corrections
must propagate through B01 wage-exposure flags, C01 normalization/merge, C02 and
affected designs. The canonical prediction cutoff and model are unchanged.
DiD's 2010 lagged-wage regressor is consequently missing. MC retains its
available-value baseline means while recording separate wage and quantile
support counts; its 2008--2009 fraction-affected value uses 2008 alone.
The border's 2008--2010 FE-only outcome calibration retains all three years.
