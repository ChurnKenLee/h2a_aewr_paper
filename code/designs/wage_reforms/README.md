# Historical wage reforms

Standalone, bounded investigation of the 2009–2010 wage-method reversal and
2015–2018 range wage phase-in. Read the
[design contract](../../../content/designs/wage-reforms.md) and `design.json`
before changing assignments, samples, outcomes, or inference.

Run from the repository root:

```sh
DRY_RUN=1 ./scripts/run_wage_reforms.sh
./scripts/run_wage_reforms.sh
UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync python -m unittest discover -s code/designs/wage_reforms/tests
```

The runner exports each Marimo application to an isolated `/tmp` directory
before execution. It makes no network calls and is outside `run_all.sh` and
the manuscript workflow.

| Producer | Inputs | Owned outputs |
| --- | --- | --- |
| `01_build.py`, with `analysis.py` | H-2A with-FIPS source; reviewed employer membership; AEWR region crosswalk; `design.json`; `policy_sources.json` | `data/processed/wage_reforms_cases.parquet`, `wage_reforms_application_cases.parquet`, `wage_reforms_business_year.parquet`; coverage, geography, timing, wage, linkage and cohort CSVs under `outputs/tables/wage_reforms/`; build manifest |
| `02_estimate.py`, with `analysis.py` | Both design panels and registry | Local wage, fixed-cohort response and state-deletion CSVs under `outputs/tables/wage_reforms/` |
| `03_report.py`, with `report.py` | Design tables and source register | `outputs/reports/wage_reforms.md`, provenance JSON, PNG/PDF figures under `outputs/figures/wage_reforms/` |
| `04_validate.py`, with `report.py` | Panels, tables, manifests and sources | Validation report; no modification of empirical results |

These are source-level microdata diagnostics. They consume the owning source
artifacts directly because the county aggregate omits individual filing dates,
job titles and wages. They do not reconstruct allocated business worker totals,
repair county panels, or replace any upstream artifact. Certified positions are
counted once per case; business outcomes use the role-aware v2 identity map.

The historical FLC archive is identified in `policy_sources.json` but has not
been recovered locally. No 2009 state-average application wage is used as a
statutory floor, and no postreform offered wage is used to assign treatment.
Response estimates are diagnostics; the report records what prevents causal
interpretation and what evidence would resolve it.

The fixed FY2011 herder cohort deliberately precedes the 2014 litigation.
Its title coverage is incomplete; `monthly_title_support.csv` retains the
generic monthly farm/ranch-animal titles. The linkage-retention and all-case
filing audits were added after the first response fits exposed large attrition
and a filing-count break; they do not change the primary fits or upstream IDs.
