+++
title = "Border data-input comparisons"
description = "Fixed labor denominators, consistent histories, source audits and unresolved external inputs."
[extra]
scopes = ["code/designs/border_discontinuity", "scripts/run_border_data_inputs.sh", "src/h2a/worksite_exposure.py", "src/h2a/worksite_geocoding.py"]
+++

The input upgrade follows the requirements of the
[bounded border redesign](@/designs/border-discontinuity.md).
`data_input_design.json` is a separate registry; it does not promote a causal
specification. Source corrections occur in A01/B01, while candidate labor
denominators and comparison samples remain design-owned.

The benchmark outcome divides annual certified positions by fixed 2011 BEA
farm jobs. The declared sensitivities use fixed means over 2008--2010 of BEA
farm jobs or hired farm jobs (farm employment less farm proprietors), and
positive observed 2007 Census hired workers. Three-year means require all
three positive observed annual values. Census totals refer to people hired
during the year; BEA totals refer to jobs. Neither is an annual full-time labor
equivalent. Different numerical standard errors or MDEs are not automatically
improvements in economically comparable precision.

Each denominator is estimated on its native eligible sample and on one common
sample with all four denominators. The comparison holds the screened h1
geography, 2011--2022 current real AEWR, county/component-year fixed effects,
equal county-year weights and certification numerator fixed. It retains pair
CRV1/CV3J and shared-endpoint dyadic diagnostics, including nonpositive raw
variances. The 2011 benchmark retains its original role; there is no selection
by coefficient, p-value, standard error or power.

Common-sample power reuses the existing four county/region/pair/spatial noise
processes, 999 simulation and calibration repetitions, seed 201103 and one
thread. FE-only calibration retains 2008--2010 outcome rows despite missing
2009 statutory wages. Invalid draws remain in the simulation denominator.
The upper-Wilson size gate is at most 0.075 with no invalid variance draws;
failed gates leave size-valid MDEs unavailable. The signed effect grid is
expressed in each denominator's units and is not an equal economic threshold
across those units. No extra independent assignments arise from a denominator
change or a finer outcome calendar.

The distinct [worksite calendar](@/contracts/h2a-worksite-exposure.md) supplies
capacity, timing and coverage diagnostics in a unique wide county-year panel.
Its nominal capacity-weighted FLS reference is not used in the regressions.
The [AEWR source contract](@/contracts/aewr-inputs.md) removes nonstatutory 2009
means from canonical rates and retains dated applicability gaps.

Available source history extends to 1990 for annual wage references, 1997 for
county Census hired labor, 2001 for cleaned BEA farm employment and 2000 for
local QCEW. Source/year/AEWR-region audits retain missing and suppressed cells.
QCEW covered jobs exclude proprietors and parts of agricultural employment.
The shared outcome backbone remains 2008--2022. Earlier missing H-2A outcomes
are not filled with zeros; later disclosed cases do not automatically extend
the treatment regime. The sourced policy-event ledger records concurrent
changes without assuming that regional wage movements are exogenous.

Non-public NASS data and custom tabulations are unavailable by user instruction.
The former NASS access proposal is withdrawn. The current county analysis uses
public BEA and Census totals; it does not need a farm-level nonparticipant frame
or institutional-access details. Application files alone cannot identify the
population of farms that never apply, which matters for a different estimand:
the probability that a farm participates. That farm-adoption design is not an
input requirement for the current county comparisons.

Historical extensions still need actual 2009 OES determinations by
area/occupation/skill/cohort, missing occupational lookup vintages and pre-2008
case/job-order archives where retained. Finer spatial extensions need worksite
coordinates with boundary/outcome-linkage audits. A separately declared
public-data alternative could measure certification capacity per predetermined
agricultural area, including zero-certification areas in its geographic frame.
It would measure utilization intensity rather than farm adoption. Missing
geocodes or administrative coverage must not be coded as zero activity; neither
this finer estimator nor a new outcome is implemented by the current registry.

The remaining archival draft concerns DOL source records only, with availability
and variable definitions left to the provider. Report generation sends no
request. Programmatic Census/FLAG downloads failed at hostname resolution in
the execution environment; this technical failure is separate from permission
to obtain non-public data and does not prevent the existing county fits.

The independent artifact validator checks source keys/hashes, 2009 missingness,
calendar and county-year conservation, denominator identities/common samples,
simulation settings, invalid-draw accounting and provenance. Passing those
checks does not complete external acquisition or establish identification.
