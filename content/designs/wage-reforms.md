+++
title = "Historical wage-reform investigation"
description = "Dated wage-method reversal and herder phase-in diagnostics, source gaps, employer cohorts and identification gates."

[extra]
scopes = ["code/designs/wage_reforms", "scripts/run_wage_reforms.sh"]
+++

This standalone branch investigates whether the 2009–2010 wage-method reversal
or 2015–2018 range wage phase-in can identify the effect of required wages on
H-2A use. It produces a bounded feasibility report, not a promoted causal
elasticity. Its source-level observations cannot be recovered from the county
panel, so it consumes the owning with-FIPS and employer-membership artifacts
directly. It does not repair upstream schemas or reconstruct business worker
allocations. The [AEWR input contract](@/contracts/aewr-inputs.md) governs the
distinction between statutory references and observed application wages.

The operational commands and output ownership are in
`code/designs/wage_reforms/README.md`. The source-traced findings are generated
in `outputs/reports/wage_reforms.md`. `design.json` declares the bounded menu;
`policy_sources.json` records identified primary sources and access gaps.
The runner is outside `run_all.sh`, the manuscript, and welfare.

## Evidence status and timing

The registry was declared after descriptive coverage showed a material share
of participating employers, but before the new response fits. Two measurement
audits were added after the first fits: alternative-tier identity reachability
in response to large apparent attrition, and all-case filing counts in response
to the deadline count break. Those additions do not change the primary sample,
assignments, estimates, or employer linkage.

The 2008 rule took effect January 17, 2009 and the 2010 rule applied from
March 15, 2010. The 2010 transition guidance also distinguishes June 1 starts;
pre-March-15 filings with such starts or invalid contract dates are omitted from
the local 2010 wage fit as ambiguous. The proposed June 2009 suspension was
enjoined and is not a third implemented switch. Older 2008-rule contracts
retained their wage principles. The source register links the
[DOL transition FAQ](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/h2a_2010_final_rule_faqs_round1.pdf)
and court chronology.

The herder national monthly reference is 1206.31 from November 16, 2015 through
2016, 1389.67 in 2017 and the corrected 1584.22 in 2018. These are wage floors,
not estimated average wages. They apply to work performed during existing
contracts as well as new ones, subject to higher statutory/CBA floors.
Filing-date grandfathering is therefore not a valid herder control design.
Fiscal years identify disclosure cohorts, not the exact start of legal exposure.
The herder analysis ends in 2018 before California's additional overtime change.

## Data, units and fixed cohorts

The source window is FY2008–2019. Positive-certified cases use the existing B01
national-audit rule: a singleton positive source row or the applicable master
flag. Every positive case-year must have exactly one selected row. Certified
positions are counted once per case; these are not unique workers or admissions,
and a joint case total is not repeated as each member business's employment.
The all-case filing audit separately includes nonpositive records with a unique
valid within-case-year receipt date. Case numbers appearing across multiple
fiscal vintages are counted and disclosed.

Dates are parsed without hand repairs. Start/end fields use the first observed
certification, job and requested date, in that priority, accommodating the
FY2016 disclosure field change. Contract dates must be in fiscal-year±1 and
end on/after start. Receipt years must be between fiscal-year−2 and fiscal-year.
Invalid and ambiguous dates are counted. No early requested positions or weekly
hours are imputed.

Wages remain nominal dollars in the reported unit. Positive numeric hourly
HOUR/HR and monthly MONTH/MTH values are analyzed separately. There is no
winsorization, deflation or monthly-to-hourly conversion. Strict herder screening
requires monthly pay plus the registry's herder/range-livestock title regex;
explicit sheep is an additional title screen for possible state-floor controls.
Neither title nor all-monthly screening proves regulatory range-duty coverage.
The entire monthly title and pay-unit quality distributions remain available.

The early cohort fixes reviewed v2 balanced worksite businesses in FY2008 and
follows FY2008–2012. The herder cohort fixes them in FY2011 and follows
FY2011–2018, with FY2014 as the response reference. Full FY2008–2024 identity
evidence follows the existing [linkage contract](@/contracts/employer-linkage.md).
Baseline rows determine pure monthly herder, pure hourly nonherder, or mixed/
unclassified type. Future outcomes never reassign type or geography. Absence of
positive certification is zero participation, not proof of business closure.
This is an incumbent-cohort estimand, not entry among all farms.

Main comparisons require a single valid own worksite state and a single state
across all associated baseline case rows. Mixed or missing evidence remains in
audits. This screen can exclude a business whose joint case contains other
members in other states, and cannot establish a complete worker itinerary.
Candidate higher-floor CA/OR/HI status is only a support category. Hawaii's
earlier prevailing wage is not assumed to be a continuing statutory floor;
California's wage path and employer-size rules need separate treatment.

The primary diagnostic outcome is any positive-certified program participation
in the fiscal year, including jobs outside the fixed baseline type. Same-type
participation is a classification diagnostic. Log median offered wage is
computed from distinct business-case-wage observations of the baseline type;
it stays missing for nonparticipants. Paired wage changes condition on survival
and observed wages. Their population differs from the participation outcome,
so their coefficients cannot be combined into a Wald elasticity.

## Estimation and inference

For each implemented cutoff and its same-calendar-date 2011/2012 comparisons,
the local offered-wage fit uses the complete declared ±30/60/90-day menu.
OLS includes post, day distance and post×distance plus state and normalized
job-title FE. Cases receive equal weight. AEWR-region CRV1 uses exact absorbed
FE rank and t(G−1) intervals. Exact title singletons remain in the rank/count
calculation. Positive-certification conditioning and manipulable seasonal
filing prevent interpreting these coefficients as causal RD estimates. Annual
wage notices can also contaminate the calendar comparisons.

For each fixed cohort, the response fit differences the outcome against its
declared reference year and regresses the difference on the baseline herder
indicator and state FE, with equal business weight. Only states with both
baseline groups in that fit contribute. Inference is state CRV1 with t(G−1);
every state deletion is retained as influence evidence. Paired wage observations
may have very few states. National sector shocks can invalidate this reference
law even when the mechanical covariance calculation succeeds.

The conditional 80% MDE is `[t(.975,G−1)+normal(.8)] × SE`. It is an approximate
precision diagnostic at two-sided alpha .05, not size-validated power or
observed power. No simulation-calibrated claim of adequate power is made.
Fixed-effect absorption is checked against a full dummy-variable OLS and direct
cluster sandwich on a synthetic crossed-FE sample. Rank, cluster and resource
failures have explicit output statuses. The resource guards are one million
source rows, 100,000 regression rows and 1,000 total design-rank columns.

{{ <grounding path="code/designs/wage_reforms/analysis.py" anchor="wage-reforms-estimation" sha256="f9bab8b9d79b53747136f6b26bdea23d9db73683446d4a2aabb615de3b1cb682" /> }}

The candidate protected-state sheep triple difference receives a support audit,
not a fitted elasticity. Several employers in one protected state do not create
several independent policy comparisons. The existing source audit and the
cohort results determine whether recovery of better assignments is worthwhile;
they do not establish parallel untreated trends.

## Promotion requirements

The reversal needs historical FLC area/occupation/skill wage vintages and
validity dates, baseline job information independent of postreform wages,
competing floors and resolved transition cohorts. The identified
[DOL archive at DataLumos](https://www.datalumos.org/datalumos/project/100447/version/V1/view)
has not been downloaded locally. Generic OES wage statistics are not substituted
for exact FLC levels; the quarantined 2009 state application means are unused.
A credible design must separate the differential wage gap from the concurrent
recruitment/enforcement changes. Predicting renewal timing from preannouncement
filings is a distinct unestimated candidate with its own seasonal exclusion
restriction, not a repair of the manipulable filing-date RD.

The herder design needs verified range duties, itinerary-specific state/CBA
floors, and reviewed identity continuity across disclosure regimes. Higher-floor
controls must have adequate independent support and their own wage changes must
be modeled. The 2017–2018 steps were preannounced; with only herders, national
phase-in variation is absorbed by herder-year effects. Apparent responses before
the actual wage increase, title changes, and linkage sensitivity are retained
as challenges to interpretation.

## Reproducibility

The build owns three design Parquets and descriptive/measurement CSVs; estimation
owns the full local, cohort and deletion menu; reporting owns Markdown, PNG/PDF
figures and provenance. The final validator checks hashes, nonempty unique keys,
canonical geography, constant baseline assignment, complete cohort years, zeros
and wage missingness, case-total arithmetic, the fit menu and the corrected
monthly wage references. Passing checks demonstrates executable consistency,
not identification. No other design's outcomes or specifications change.

{{ <grounding path="scripts/run_wage_reforms.sh" anchor="wage-reforms-runner" sha256="2376f5f994d0f0b40667ec43c53b1185fe740d6a911eadde81e8b4eeb11bfce5" /> }}
