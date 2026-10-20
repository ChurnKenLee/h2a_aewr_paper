+++
title = "H-2A allocation and worksite calendar"
description = "Certification capacity, source dates, geographic uncertainty and allocation conservation."
[extra]
scopes = ["code/b01_derived/01_h2a_aggregation_nodupes.R", "code/b01_derived/01b_h2a_worksite_exposure.py", "src/h2a/worksite_exposure.py", "code/a01_sources/02_05_h2a_geocode_worksites.py", "src/h2a/worksite_geocoding.py", "code/designs/border_discontinuity", "scripts/run_border_data_inputs.sh"]
+++

B01 preserves case totals and the existing proportional/equal worksite
allocation before case-modal dates, date exclusions, hours imputation or county
splits. `allocation_id` records the owning source dataset and stable source row.
Raw requested staffing, site/case dates, source hours, allocated certified
positions and the allocation method remain available for audit.

{{ <grounding path="code/b01_derived/01_h2a_aggregation_nodupes.R" anchor="h2a-worksite-allocation-audit" sha256="f29d46f79e5382af639730cc926b60d678d930f6a30b81a3dfceab9ccd45beca" /> }}

The separate B01 calendar producer reconciles allocated positions against the
case total before expansion. Nonfinite, negative or unreconciled allocations
are quarantined with a case audit. It does not repair the historical benchmark
or infer which duplicate administrative record is correct.

The measure is **allocated certification position-days**. Site requested
begin/end dates are inclusive. Valid site intervals take precedence; otherwise
the producer labels a valid case-date fallback. Begin and end years must lie
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
