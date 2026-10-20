+++
title = "H-2A worksite-business linkage"
description = "Role-aware FY2008--2024 business universe, nested linkage rules, review protocol, and downstream count semantics."

[extra]
scopes = ["code/a01_sources/02_04_h2a_match_employers.py", "src/h2a/employer_linkage.py", "code/b01_derived/01_h2a_aggregation_nodupes.R", "code/c01_clean", "code/c02_build", "code/designs"]
+++

The supported H-2A identity unit is a worksite business, not a filing agent,
association headquarters, contractor, application, worksite row, or generic
national name. Version `v2` uses all available identity evidence from
FY2008--2024. The canonical partition is therefore a full-vintage longitudinal
linkage; a separate audit reconstructs the FY2008--2022 partition using only
evidence available through FY2022.

{{ <grounding path="src/h2a/employer_linkage.py" anchor="employer-linkage-contract" sha256="ae782b1010011facde5da0dd95b87e935f49dd0704d43cae3291bc484cfa407e" /> }}

## Outcome-free business universe

Role classification uses source, fiscal year, case membership, the historical
primary/sub indicator, organization and application-role fields, the
association/agency field, the contractor flag, and the presence of a named
Addendum B business. It never consults status, requested or certified workers,
hours, wages, dates, or downstream outcomes.

Before FY2020, standalone rows and sub-employer rows are businesses. A master
row is retained only when the source role identifies a substantive employer;
association, agent, and contractor masters are excluded. From FY2020 onward,
each named Addendum B row is the business. A master is the business only when
the case has no named Addendum B business. An individual-employer master in a
case with named Addendum B businesses is alias evidence only: it can support a
same-case balanced link but is never counted as another business. Other
masters with named Addendum B businesses are excluded from the matching
universe.

The source-row membership artifact records these roles and the inclusion and
alias-only flags. Every named included record must receive all three linkage
IDs. Blank Addendum names remain explicitly excluded rather than receiving
fabricated identities.

## Nested deterministic partitions

All name evidence is ASCII-folded, upper-cased, punctuation-normalized, and
space-normalized. ZIP codes, states, phones, FEINs, and full addresses have
separate normalizers. Placeholder aliases and names containing no distinctive
core token cannot supply name evidence. A DBA used by more than ten distinct
primary names is excluded. An exact address or phone used by more than ten
distinct valid names is a shared administrative contact and cannot corroborate
a link.

The tiers are nested:

- Conservative accepts exact FEIN, plus exact primary-name or valid-DBA links
  corroborated by an eligible exact full address or phone.
- Balanced adds name similarity of at least 97 with an eligible exact address
  or phone; exact distinctive name within state and ZIP; similarity of at
  least 98 within state and ZIP with a shared distinctive core token; and an
  individual-master/Addendum B same-case bridge with similarity of at least
  97. It never accepts national exact-name-only evidence.
- High recall adds exact distinctive-name links across ZIP codes within a
  state and the broader same-state, shared-core-token fuzzy rule at similarity
  of at least 94.

No union may combine conflicting nonblank FEINs. Non-FEIN unions also obey
distinct-name sprawl limits of 25, 100, and 500 for conservative, balanced,
and high recall. Candidate order, union anchors, review ranking, and output
sorts are deterministic. IDs have the form
`emp_wb_v2_<tier>_<anchor-hash>`.

## Audits and manual review

The A01 producer owns:

- `h2a_employer_crosswalk.parquet`, the canonical raw-identity crosswalk;
- `h2a_employer_record_membership.parquet`, the case/year source role map;
- `h2a_employer_edge_audit.parquet`, accepted edges and structural or
  near-threshold rejections with scores;
- `h2a_employer_cluster_diagnostics.parquet`, cluster size, name, geography,
  FEIN, era, and transitivity diagnostics;
- `h2a_employer_linkage_vintage_comparison.parquet`, the full-vintage versus
  as-of-2022 partition comparison; and
- the deterministic review sample and report.

Seed 1729 selects exactly 200 review pairs: 140 accepted balanced edges
stratified by rule and data regime, 40 near-threshold rejected candidates, and
20 balanced-cluster anchor-to-remote-member checks. Review strata oversample
cross-era, cross-state, multi-state, common-name, and large-cluster cases and
retain inverse sampling weights.

Manual labels live at
`data/raw/h2a/employer_linkage_review_labels.csv` and contain one
`same_business = true|false|uncertain` value for every sampled `pair_id`.
Reviewers identify the business by the entity paying the workers. A consulting
or farm-management firm and its client are not the same business merely because
they share management or contact information; the firm is the relevant business
only when it is the worker-paying entity represented by the record.
`uncertain` counts as a false link in weighted precision. The report gives
overall, rule, and regime precision, near-miss yield, and cluster-check failure
rates. Those empirical metrics are diagnostic and do not select or promote a
tier.

Promotion still fails if labels are missing or incomplete, IDs or keys are
missing or duplicated, tiers are not nested, IDs change after a row shuffle,
a cluster contains conflicting FEINs, or a record role is invalid. The review
report hash and status travel with the linkage metadata.

## County-year interface

The existing `employer_id_{conservative,balanced,high_recall}` and
`nbr_employers_{conservative,balanced,high_recall}_start_year` names are
retained, but their old semantics are superseded. B01 counts distinct linked
worksite businesses with a positive allocated certified-worker quantity in a
county and contract start year. It does not count alias-only or excluded
masters. All worker, hour, wage, application, case-status, and geographic
allocation calculations remain unchanged.

The aggregate and shared panels carry one filled contract:
`employer_linkage_unit = worksite_business`,
`employer_linkage_version = v2`, input maximum year 2024, review status
`complete`, and the 64-character review-report hash. Balanced linkage is the
canonical causal outcome tier; conservative and high-recall counts remain
sensitivity measures.
