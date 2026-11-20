+++
title = "H-2A allocation and worksite calendar"
description = "Certification capacity, source dates, geographic uncertainty and allocation conservation."
[extra]
scopes = ["code/a01_sources/02_01_h2a_match_locations.py", "code/b01_derived/01_h2a_aggregation_nodupes.R", "code/b01_derived/01b_h2a_worksite_exposure.py", "src/h2a/worksite_exposure.py", "code/a01_sources/02_05_h2a_geocode_worksites.py", "src/h2a/worksite_geocoding.py", "code/designs/border_discontinuity", "scripts/run_border_data_inputs.sh"]
+++

## Employment-period dates

A01/02_01 owns the mapping directly in its `disclosure_date_helpers` cell.
Consumers never use `JOB_START_DATE` or `JOB_END_DATE` as employment dates. Raw certification,
requested, employment and job-order dates remain separate strings. The
canonical period is selected as a pair:

| Fiscal years | Primary source pair | `period_date_basis` |
| --- | --- | --- |
| 2008--2015 | `CERTIFICATION_BEGIN_DATE`, `CERTIFICATION_END_DATE` | `certification_dates` |
| 2016--2017 | `REQUESTED_START_DATE_OF_NEED`, `REQUESTED_END_DATE_OF_NEED` | `legacy_need_dates` |
| 2018--2019 | Same dates-of-need pair | `granted_need_dates` |
| 2020--2025 | `EMPLOYMENT_BEGIN_DATE`, `EMPLOYMENT_END_DATE` | `employment_dates` |

The cached DOL layouts under `data/raw/h2a/documentation` distinguish job-order
dates from employment periods. FY2016--2017 layouts still label certification
fields absent from the delivered workbooks, so their date-of-need semantics
remain explicitly legacy. FY2018--2019 layouts describe their dates-of-need
fields as granted dates. From FY2020, requested and certified employment
periods are separately named. Hill--Javier instead selects requested dates in
2009--2014 and 2020--2024; that alternative is reconstructible from raw fields.

A valid period has parseable dates, end on/after begin, and both endpoint years
within fiscal-year ±1. When the primary pair is invalid, a complete valid
requested pair is a `requested_fallback`; endpoints are never mixed. If only
a plausible start is available, retain it with a `_start_only` basis and a
null end. Otherwise both dates are null and the basis is `missing`. These
source dates receive no case-modal replacement or manual typo correction.
This changes the earlier benchmark's date cleaning as well as correcting
FY2016--2019 job-order timing. Start-only rows count toward start-year positions
and linked businesses; hours and calendar allocation require complete periods.
The existing benchmark hours formula uses end minus begin, while the separate
monthly capacity measure uses inclusive days; this change does not alter those
duration conventions or the benchmark's pooled weekly-hours imputation.

`application_begin_date` prefers a plausible requested start for every status,
otherwise the canonical period start; `application_date_basis` records that
choice. B01 uses this one cohort for application/status counts and retains its
labeled-in-code fiscal-year fallback when no start is available. Certification
rates thus share an application-cohort denominator, while positions per
application can compare employment-start positions with intended-start filings.
Receipt and decision dates remain separate administrative timing fields.

{{ <grounding path="code/a01_sources/02_01_h2a_match_locations.py" anchor="h2a-disclosure-dates" sha256="2a37703958f14487907971421af80c35debe4286cb212732a8a3a2d00a239e9a" /> }}

## Allocation and calendar

B01 preserves case totals and the existing proportional/equal worksite
allocation before date exclusions, hours imputation or county splits.
`allocation_id` records the owning source dataset and stable source row.
Raw requested staffing, site/case dates, canonical period dates and basis,
source hours, allocated certified positions and allocation method remain
available for audit.

{{ <grounding path="code/b01_derived/01_h2a_aggregation_nodupes.R" anchor="h2a-worksite-allocation-audit" sha256="37c0c46a9ed06aeca1a447be0c5fdb978585004f426b4155f24eb16d43780599" /> }}

The separate B01 calendar producer reconciles allocated positions against the
case total before expansion. Nonfinite, negative or unreconciled allocations
are quarantined with a case audit. It does not repair the county benchmark
or infer which duplicate administrative record is correct.

The measure is **allocated certification position-days**. Site requested
begin/end dates are inclusive. Valid site intervals take precedence; otherwise
the producer labels a valid canonical case-period fallback and retains
`period_date_basis` in monthly rows. Begin and end years must lie
within one year of the source fiscal year. There is no modal-date or typo
correction. Before FY2020, separately observed site-date fields are unavailable.
Requested staffing weights are not realized certified staffing: a crew may
rotate among sequential worksites. These measures do not count visa entries,
distinct people, actual days worked or paid employment.

Distinct candidate counties receive equal shares. An unresolved county remains
null in the monthly artifact and contributes to its unmatched-coverage audit;
it is excluded from county aggregates. Geographic identifiers use the shared
[string/2010-vintage contract](@/contracts/geographic-codes.md).
Within each fiscal partition the key is `allocation_id, county_fips, month`;
the reusable annual aggregate is unique by `county_fips, year`. Calendar
expansion conserves allocated position-days, including unmatched counties.

Reported or case-inherited weekly hours contribute position-days times
hours/7 only when finite and in (0,168]. Missing hours stay null, including in
an aggregate with no observed schedule. The annual output separately records
position-days with observed site dates, case fallback, ambiguous county splits,
imputed staffing and reported hours. The strict site-date component stays null
when no site-date observation exists; this does not mean zero exposure.

Dated FLS references join on actual interval overlap. Uncovered days remain
in the coverage denominator, and the 2009--early-2010 gap is preserved.
These are [reference wages](@/contracts/aewr-inputs.md), conditional on
unresolved occupation, cohort and competing-floor applicability. The
capacity-weighted nominal reference uses outcome-dependent weights; it is a
timing diagnostic and cannot be treated as predetermined policy or an instrument.

The point-geocoding producer uses actual worksite streets from Addendum B and
the main FY2020--2024 disclosure. It never substitutes filing-agent/employer
mailing addresses. Addendum source-row IDs support linkage to allocation
records. Main disclosure addresses retain an unresolved case/worksite linkage
flag. No point is assigned until a complete provider response is validated.
Address-range matches still require quality, 2010 boundary-vintage, near-boundary
ambiguity and outcome-linkage review. The moving Census benchmark/vintage and
its raw returned county code are retained separately from canonical county IDs.
A county centroid is not a substitute for this work.

The calendar and applicant addresses do not enumerate farms that never apply.
Those farms would be needed for a farm-level participation probability; the
current county analysis instead uses public labor totals and does not require
farm microdata. Non-public NASS data/custom tabulations are unavailable and
are not an access route for this project. A different public-data spatial
estimand could normalize allocated certifications by predetermined agricultural
area, using a complete geographic frame including zero-certification areas.
That would measure utilization intensity, require explicit outcome/coverage
rules and geocoding audits, and would not identify farm adoption probabilities.
