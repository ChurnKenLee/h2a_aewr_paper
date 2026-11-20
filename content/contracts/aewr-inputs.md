+++
title = "AEWR input history and applicability"
description = "Source-correct annual references, dated wage lookup tables and explicit coverage gaps."
[extra]
scopes = ["code/a01_sources/01_aewr_extract_tables.py", "src/h2a/aewr.py", "code/b01_derived/01_h2a_aggregation_nodupes.R", "code/c01_clean", "code/designs"]
+++

The source producer preserves annual FLS reference rates for 49 states from
1990 through 2025. Rates through 2008 come from CRS Table 1; later rates come
from explicitly selected Federal Register state tables. The USDA workbook is
an independent audit and never selects the output sample. The Hawaii 2003
correction is sourced to document 03-6559.

{{ <grounding path="code/a01_sources/01_aewr_extract_tables.py" anchor="aewr-source-contract" sha256="10a61361c3bf22179d84329fb283eced41c86e2724d8e236f4076e733207fdc1" /> }}

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
end date. Missing lookup cells are not zero.

The preceding 2023--2025 other-occupation workbooks are cached under
`data/raw/aewr_2023_oews_occupation/flag`, a folder separate from the AEWR raw
folder. `aewr_oews_occupation_rates.parquet` holds the 2023-rule rate for every
occupation outside the six common field-and-livestock SOCs, by state and OEWS
vintage: the statewide OEWS mean hourly wage, or the national mean when no
statewide wage is published. It covers four vintages effective from 2023-03-30
until the rule's vacatur was implemented on 2025-08-29, with a later start for
Kansas-order states and entities in the last vintage. Top-coded and unpublished
cells stay null with an explicit `rate_basis`. The OEWS-based field rates of
AK, DC, GU, PR and VI are not in this table. Its workbook hashes are recorded
under `occupation_inputs` in the manifest, apart from `inputs`.

B01 uses these tables only to classify wage offers. An hourly offer counts as
AEWR-exposed if it is within five cents of the annual reference for the
contract start year or the year before, or of a dated occupation or U.S.-worker
skill-tier rate that applied to the job's occupation at any time in the year
ending on the contract start date. The occupation is the SOC code on the case
record, because Addendum B worksite rows carry none. No dated rate exists
before 2023-03-30, so earlier years rest on the annual rule alone and their
values did not change when the dated rates were added. The classification does
not resolve combined-occupation duties, multi-state itineraries or injunction
membership. Contracts starting after 2025-10-02 fall under the skill-tier
rule, where fewer offers sit at an AEWR, so the 2025 share is lower for that
reason and not through misclassification.

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
