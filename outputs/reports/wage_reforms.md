# Historical wage reforms: investigation and decision

The **2009–2010 reversal remains the stronger identification candidate**, provided
its local occupation/skill wage schedules can be recovered. The 2010 reversal
has a visible offered-wage response. A filing-date RD alone is not credible
because filing composition changes around the deadlines and the rules also
changed recruitment and enforcement.

The **herder phase-in has a large wage first stage but currently lacks a convincing
comparison group**. Its apparent employer-participation response mostly precedes
the wage increase, and its employer-retention measurement is sensitive to the
linkage tier. Neither investigation presently supports a causal wage elasticity.
These are findings of the bounded investigation, not a preference selected by
statistical significance.

## 1. The reversal: price response and sorting

The 2008 rule became effective January 17, 2009; the 2010 rule applied from
March 15, 2010. The latter reinstated FLS wages for new-rule applications while
old-rule contracts retained their wage principles. Cases filed before March 15
with starts on/after June 1 are kept out of the 2010 local fit because the
transition guidance makes their classification ambiguous. The attempted June
2009 suspension was enjoined and is not an implemented third switch.
[DOL transition guidance](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/h2a_2010_final_rule_faqs_round1.pdf); [court order](https://docs.justia.com/cases/federal/district-courts/north-carolina/ncmdce/1%3A2009cv00411/51449/57).

The following local linear fits use **offered hourly wages of positive-certified
cases**, with state and normalized job-title fixed effects, separate slopes on
each side, equal case weight and AEWR-region CRV1 inference. Intervals use
t(G−1). Reported percentage changes and interval endpoints exponentiate the log
coefficients. All three windows were declared before fitting; none is promoted
as a causal specification.

| Cutoff | Window | Cases | AEWR regions | Wage discontinuity | Conditional 95% interval |
| --- | --- | --- | --- | --- | --- |
| 2009-01-17 | ±30 days | 1386 | 17 | -4.4% | [-14.0%, +6.2%] |
| 2009-01-17 | ±60 days | 2610 | 17 | -1.6% | [-7.2%, +4.3%] |
| 2009-01-17 | ±90 days | 3363 | 17 | -2.6% | [-7.7%, +2.7%] |
| 2010-03-15 | ±30 days | 832 | 16 | +9.1% | [-0.1%, +19.2%] |
| 2010-03-15 | ±60 days | 1809 | 17 | +8.7% | [+1.1%, +16.8%] |
| 2010-03-15 | ±90 days | 2551 | 17 | +9.9% | [+1.8%, +18.5%] |

The weekly all-case disclosure count around March 15, 2010 falls from
**268 in the preceding week to 35 in the following week**.
Among positive-certified cases the corresponding counts are
263 and 30. The all-case
audit includes denied/withdrawn/other zero-certified records; the wage sample
conditions on positive certification. These count changes are a sorting and
composition diagnostic, not a formal density test or a causal estimate of
reduced application demand. Same-calendar-date 2011/2012 wage and count
comparisons are retained in `local_wage_responses.csv` and the figure. Annual
AEWR notices can affect these comparison windows too; they are not guaranteed
untreated placebos.

![Wages and filings around the reversal](../figures/wage_reforms/reversal_cutoffs.png)

Early SOC and requested-position fields are absent. The source has job titles
and offered wages, but a generic title does not establish a skill level. The
local regression is consequently a **wage-response diagnostic, not an IV first
stage for a verified statutory exposure**. The 2009 state application-average
wages are not legal rates and are not used anywhere in this branch.
[2010 final rule](https://www.govinfo.gov/content/pkg/FR-2010-02-12/pdf/2010-2731.pdf).

The next defensible design is a **continuous-exposure reversal**: freeze each
baseline job's geography and occupation/skill, construct the gap between its
counterfactual FLS and FLC schedules, and interact that independently assigned
gap with the two implemented regimes. Use fixed employer cohorts with zeros,
plus market-by-job cells for entrants, and distinguish incumbent return from
entry among all potential farms. Common national reform effects can be absorbed;
identification still requires bundled nonwage changes not to have differential
effects correlated with the wage gap. A return to the old wage method need not
undo entry sunk costs immediately.

The historical FLC archive is identified, but **its files have not been recovered
locally**. The official current archive only reaches 2021–2022. The DataLumos
collection identifies original DOL wage files back to FY2007, including the
geographic and occupational tables. Folder/download access failed here, and
shell network access could not resolve the host. The local BLS OES archives
contain wage statistics, not verified FLC Levels I–IV. No substituted schedule
or assigned skill has been fabricated.
[historical DOL data archive](https://www.datalumos.org/datalumos/project/100447/version/V1/view); [official downloads](https://flag.dol.gov/wage-data/wage-data-downloads).

A second, more speculative route is **predicted renewal timing**: use an
employer's preannouncement filing or crop-calendar pattern to predict whether
its next application would fall under the old or new wage rule. This avoids
using its chosen postannouncement filing date as assignment. It would still
need a strong timing first stage and calendar-seasonality comparisons; this
investigation does not assume that predictable seasonality is random assignment.

## 2. Herders: large wage changes, thin unaffected support

The national monthly floor is $1,206.31 from November 16, 2015 through 2016,
$1,389.67 in 2017, and $1,584.22 in 2018. The last figure incorporates DOL's
correction to an inconsistent printed notice. Higher applicable state or CBA
floors continue to bind, and wage increases also apply during existing
contracts. **Earlier-filed herder contracts are not an untreated grandfathered
group.** The 2015 rule retained the earlier 364-day sheep/goat and ten-month
other-livestock need limits; those are distinct from visa duration.
[initial floor](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/Herder_AEWR.pdf); [2017 floor](https://www.govinfo.gov/content/pkg/FR-2016-12-23/pdf/2016-30923.pdf);
[2018 correction](https://www.dol.gov/agencies/eta/foreign-labor/news?lang=ar); [ongoing-contract guidance](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/herder_faq_round2.pdf);
[2015 rule](https://www.govinfo.gov/content/pkg/FR-2015-10-16/pdf/2015-26252.pdf).

Monthly pay plus an explicit herder/range-livestock title is the strict screening
definition. It is not verified legal coverage. All monthly-pay cases and their
titles are retained separately. In particular, FY2011–2012 often use generic
farm-and-ranch-animal titles, so apparent changes in title-screened coverage do
not establish changes in the regulatory population.

| FY | Herder positions | % of program positions | Herder businesses | % of program businesses | Herder cases | Median monthly offer |
| --- | --- | --- | --- | --- | --- | --- |
| 2013 | 2,180 | 2.21% | 609 | 10.12% | 314 | $750.00 |
| 2014 | 2,532 | 2.17% | 532 | 8.49% | 385 | $750.00 |
| 2015 | 2,498 | 1.79% | 590 | 8.77% | 389 | $750.00 |
| 2016 | 2,386 | 1.44% | 469 | 6.64% | 631 | $1,206.31 |
| 2017 | 2,409 | 1.20% | 455 | 6.03% | 924 | $1,389.67 |
| 2018 | 2,255 | 0.93% | 470 | 5.66% | 895 | $1,584.22 |

Positions are **certified positions, not unique workers or visas**. Cases are
counted once using the upstream national-audit singleton/master rule. Employer
counts use distinct reviewed v2 balanced worksite businesses on positive source
rows. Case counts increase substantially during the phase-in while positions
are much steadier: a case-count outcome can reflect filing organization as well
as labor demand. No master total is assigned in full to each member business.

The FY2014 geography audit partitions the screened businesses as follows:

| FY2014 observed geography screen | Herder businesses | Explicit sheep titles |
| --- | --- | --- |
| missing_state_evidence | 2 | 1 |
| multiple_associated_case_states | 114 | 113 |
| single_candidate_protected_state | 65 | 62 |
| single_other_state | 351 | 227 |

Associated-case states include every observed row on the business's baseline
cases. This is deliberately conservative: a joint case may list other members'
states without proving that the same worker travels to each state. Conversely,
a single observed state does not prove a complete itinerary. Reported low
monthly wages in higher-floor states cannot be converted directly into wage
exposure.

For an explicit-sheep triple difference, the fixed FY2011 cohort with the strict
geography screen has **7 candidate protected employers across
1 state(s)** (CA: 7). The proposed California/
Oregon/Hawaii comparison therefore does not provide many independent untreated
policy assignments. Hawaii's earlier prevailing wage is not assumed to remain
a statutory floor. California's floor changes independently and depends on
total employer size, which certified H-2A positions do not measure. The response
window ends in 2018 before California's additional 2019 overtime phase-in.
[California wage order](https://www.dir.ca.gov/t8/11140.html); [overtime history](https://www.dir.ca.gov/dlse/Sheepherders-owed-as-a-result-of-AB-1066s-Overtime-Phase-In.html).

## 3. Employer response: the pre-period is a problem

The herder diagnostic fixes baseline business type and geography in FY2011.
It follows all those businesses through FY2018, including zeros when no
positive certification is observed. The outcome counts any subsequent program
use, even if the business changes job type. Mixed baseline types and ambiguous
geographies remain in the audit. The response contrasts changes for herder and
hourly nonherder businesses within the same baseline state, relative to FY2014.
It is an incumbent-cohort comparison; it is not entry among all farms, and
absence from a disclosure year is not proof that a business closed.

| FY | Participation contrast vs FY2014 | Conditional 95% interval | Herder firms | Hourly firms | States | Conditional MDE |
| --- | --- | --- | --- | --- | --- | --- |
| 2012 | -10.3 pp | [-19.4, -1.2] pp | 183 | 947 | 11 | 12.6 pp |
| 2013 | -2.7 pp | [-9.2, +3.7] pp | 183 | 947 | 11 | 8.9 pp |
| 2015 | -15.6 pp | [-23.7, -7.5] pp | 183 | 947 | 11 | 11.2 pp |
| 2016 | -18.1 pp | [-29.1, -7.1] pp | 183 | 947 | 11 | 15.1 pp |
| 2017 | -16.6 pp | [-27.3, -5.8] pp | 183 | 947 | 11 | 14.8 pp |
| 2018 | -15.5 pp | [-25.0, -5.9] pp | 183 | 947 | 11 | 13.2 pp |

The contrast is already **-15.6 percentage points in FY2015**,
which ends September 30, before the November 16 wage increase. By FY2018 it is
-15.5 points. Thus the apparent post-2015 difference
largely inherits an earlier decline. Anticipation of the rule or the June 2014
litigation is possible, but cannot be distinguished here from sector trends,
other rule provisions, or measurement changes. FY2014 is not asserted to be an
uncontaminated pre-period. [Mendoza litigation](https://law.justia.com/cases/federal/appellate-courts/cadc/13-5118/13-5118-2014-06-13.html).

![Herder wage and participation evidence](../figures/wage_reforms/herder_phase_in.png)

The original fixed cohort contains 183 geographically
supported strict herder businesses. By FY2018, 26
have a positive-certified return under the balanced IDs. Allowing a return
through any baseline-linked high-recall ID reaches
**59**; the conservative tier reaches
5. These are alternative identity
reachability diagnostics for the **same baseline entities**, not corrected
business counts or interchangeable causal estimates. Exact nonempty raw
name/address/state/postal matching flags
0 unlinked return candidates
for this cell. That narrow zero does not validate all apparent exits. The full
linkage-retention table remains alongside the results.

Wage-change regressions condition on observing wages in both years and the
same baseline pay/title type. For FY2018, the geographically comparable wage
sample contains only 22 herder businesses and
29 hourly businesses in 2 states.
These survivor comparisons are not population first stages and do not justify
forming a Wald elasticity from the participation and wage coefficients.

## 4. Precision and what would change the decision

The conditional MDE column is
[t(.975, G−1) + normal(.8)] × CRV1 standard error for 80% power at two-sided
5%. It is a local approximation under the stated error reference, **not a
size-validated power simulation or an observed-power calculation**. State
clustering also does not absorb a common national sheep-industry shock. The
FY2018 participation coefficient ranges from
-17.5 to -12.0
percentage points when each state is deleted in turn. These deletions are
influence checks, not a choice of preferred sample.

For the reversal, the priority data are historical FLC wage vintages covering
2008–2010, their area/county definitions and occupation/skill schedules, and
baseline job descriptions or wage determinations that resolve occupation and
skill without using postreform wages. Exposure must also respect recruitment
vintage, competing floors and transition cohorts. The present data establish
why that recovery is worth pursuing; they do not supply its missing exclusion
restriction.

For herders, the priority is a reviewed employer/contract panel covering the
2013–2018 disclosure-format transitions, verified range duties and itineraries,
and actual state/CBA obligations. A larger verified high-floor comparison
group could revive the triple difference. If such a group cannot be recovered,
the phase-in mostly supplies national herder time variation: the 2017 and 2018
steps were announced in 2015, and adding herder-by-year controls absorbs that
variation. Bigger wage changes alone do not solve that identification problem.

**Current decision:** retain the reversal as the lead candidate for source
recovery; retain the herder wage/coverage evidence but do not promote the simple
herder DiD or thin protected-state triple difference to a causal specification.

## Reproduction and audit trail

Run `./scripts/run_wage_reforms.sh` from the repository root. The runner exports
the four Marimo applications before execution and uses local inputs only.
The exact specification is `code/designs/wage_reforms/design.json`; sourced
policy facts and archive-access limits are in `policy_sources.json`. Source
files, other designs, the shared panel, the manuscript and welfare are not
rewritten by this runner.

The build contains 85,466
positive-certified case-years and
89,580 total source case-years.
There are 7
all-case rows without a unique valid receipt date and
12 case numbers
appearing in multiple positive fiscal vintages. Fiscal-year/case is the declared
source key; no claim of globally unique worker or application events is made.
Raw dates are parsed without hand repairs; the certification/job/requested
date-field priority accommodates the FY2016 disclosure field change. Missing
hours and requested positions remain missing.

All windows, placebo fits, failed-fit statuses, cohort paths, title/wage quality,
geography support, linkage reachability, and state deletions are retained in
`outputs/tables/wage_reforms/`. PNG and PDF figures are in
`outputs/figures/wage_reforms/`. The build and report provenance JSONs record
SHA-256 hashes, and the final validation checks source freshness, unique keys,
fixed assignments, complete cohort years, zero/wage semantics, arithmetic and
the complete declared fit menu. Successful validation establishes reproducibility,
not causal identification.
