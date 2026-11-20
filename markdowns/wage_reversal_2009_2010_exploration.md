# The 2009–2010 wage-method reversal: exploration

Scratch exploration, not a pipeline product. Scripts and outputs are in
`/tmp/claude-1000/-home-ken-Dropbox-projects-H-2A-Paper/d743ee46-cc2e-499d-91ae-ff58e9272844/scratchpad/reversal/`
(`01`–`07`, cited below as `[0x]`). Nothing in the repository was changed
except this note.

## Bottom line

- **The reversal moved wages.** Under the 2008 rule, 58% of hourly cases
  offered less than the applicable FLS AEWR, against 2% before and 6% after [03].
  The certified-weighted offered wage was 12% below the FLS rate [03].
- **The bite varied within regions.** Most of the variation in the
  local-floor proxy is within AEWR region (60%), and offered wages track the
  proxy only while the 2008 rule applied [05]. That is the variation the
  region-level designs lack.
- **Legal uncertainty shows up in behavior at every boundary.** Filings bunch
  before each rule date, about 12% of post-June-29, 2009 cases use the
  enjoined FLS schedule, and 78% of FY2010 certified positions were filed
  under the 2008 rule [02, 03, 07].
- **Treated and control periods are blurred.** The 2009 cut was expected to
  be temporary from May 2009 onward. Its response is not a steady-state
  elasticity.
- **The county response is not credible yet.** High-exposure counties grow
  steadily through 2013, after floors were restored, which looks like a
  differential trend rather than a response to a temporary cut [06]. There is
  only one pre-period year (FY2008).
- **Recommendation:** use the reversal as a case-level first-stage and
  pass-through exhibit, and as a short-run employer response with explicit
  timing. Do not use it as the paper's main elasticity until two blockers are
  resolved: the verified FLC Level I schedules, and FY2006–07 disclosure data
  for pre-trends.

## 1. Legal chronology (local sources only)

| Date | Event | Local source |
|---|---|---|
| 2008-02-13 | NPRM for the 2008 rule (73 FR 8538) | `data/raw/aewr/xml/E9-12434.xml` |
| 2008-02-26 | 2008 FLS AEWRs effective | `E8-3567.xml` |
| 2008-12-18 | 2008 final rule published (73 FR 77110) | `E9-12434.xml`, `2010-3078.xml`, `policy_sources.json` |
| 2009-01-17 | 2008 rule effective: OES area/occupation/skill AEWRs replace FLS | same; DOL's own Table 1 splits at applications "received after January 19, 2009" (`2010-2731.xml`) |
| Spring 2009 | Proposed suspension of the 2008 rule | **Unverified locally** (date not in local files) |
| 2009-05-29 | Final Suspension Rule published: suspends the 2008 rule for 9 months, reinstates the pre-2009 rules | `E9-12434.xml` |
| 2009-06-29 | Suspension effective date; 2009 FLS AEWRs announced for applications filed after June 29 | `E9-12434.xml` |
| 2009 (date unverified) | Suspension enjoined (M.D.N.C. 1:09-cv-00411); 2008 rule continues | `aewr_policy_events.parquet`, `policy_sources.json`; order date **unverified locally** |
| 2009-09-04 | NPRM restoring FLS (74 FR 45906) | `2010-2731.xml` |
| 2010-02-12 | 2010 final rule published | `2010-2731` (govinfo URL date in `policy_sources.json`) |
| 2010-03-15 | 2010 rule and 2010 FLS AEWRs effective for applications filed on/after | `2010-2731.xml`, `2010-3078.xml` |
| 2010-06-01 | Transition: normal-processing boundary; 2008-rule contracts keep old wage principles | FAQ summary in `policy_sources.json` (FAQ itself not local) |

The 2010 rule (`2010-2731.xml`) also reports:
- **Skill level choice:** 73% of 2009 applicants chose the lowest skill level,
  "the wage earned by the lowest paid 16 percent" of OES observations.
- **Other floors bound:** 60% of 2008-rule applications were certified above
  the AEWR because a prevailing or minimum wage was higher.

The state schedules (2008 FLS, enjoined 2009 FLS, 2010 FLS) and the 2010
rule's Table 1 of state average certified wages are parsed in [01].

## 2. Behavior around the events

| Window | Measure | Value | Script |
|---|---|---|---|
| Week of 2009-01-12, vs ~150/week in Dec 2008 | All-case filings before the 2008 rule took effect | 517 | [02] |
| Week of 2009-06-22, vs ~55/week in May–June 2009 | Filings before the suspension date | 112 | [02] |
| Week of 2010-03-08 → week of 2010-03-15 | Filings around the 2010 rule | 268 → 35 | [02] |
| Mar 15–Apr 11 vs Feb 15–Mar 14 filings | Ratio: 2008 / 2009 / 2010 / 2011 | 0.48 / 0.78 / 0.15 / 0.52 | [02] |
| Hourly cases, receipts Mar 2008–Mar 2011 | Share offered below the applicable FLS AEWR: 1987 / 2008 / 2010 rule | 2.1% / 58.4% / 6.2% | [03] |
| same | Share exactly at the FLS AEWR | 90.5% / 14.3% / 82.9% | [03] |
| same | Certified-weighted mean log(offered / FLS) | 0.001 / −0.122 / 0.001 | [03] |
| Apr 1–Jun 21 vs Jun 29–Sep 30, 2009 | Share at the enjoined 2009 FLS rate | 0.5% → 11.9% | [07] |
| FY2009 / FY2010 certified positions | Share filed under the 2008 rule | 63% / 78% | [03] |
| 2008-rule cases starting on/after 2010-03-15 | Grandfathered work | 1,576 cases, 32,786 positions | [03] |

Figure: `reversal_timing.png` [02]. The "below 2008 FLS" share before March
2008 reflects the lower 2007 AEWR then in force.

What the legal uncertainty does to the design:
- **Selection into filing timing.** Employers moved filings to the side of
  each boundary that suited them: the January 2009 spike, the June 2009 spike
  and the March 2010 spike. The filing date is therefore a choice, which rules
  out a filing-date RD. It also means outcomes measured by filing or fiscal
  year mix regimes.
- **Partial compliance with the enjoined suspension.** From June 29, 2009,
  about 12% of cases offered the 2009 FLS rate, against 0.5% before. Treatment
  was not uniform even within the 2008-rule period.
- **Expected reversal.** After May 29, 2009 employers knew the cut might end.
  After September 4 a return to FLS was formally proposed. Employers deciding
  whether to enter the program, which has fixed costs, faced a temporary price
  cut, so responses should be smaller than for a permanent change.
- **Grandfathering.** FY2010 is mostly 2008-rule work. The clean "restored"
  period starts with FY2011 filings.

## 3. Exposure: a proxy for the 2009 local floor

The verified FLC Level I–IV schedules are still missing (`policy_sources.json`:
the DataLumos archive was not retrieved). The proxy is therefore approximate:
- **Local floor proxy:** the interpolated 17th percentile of the OEWS
  45-2092 hourly wage in the county's OEWS area. Area values cover 76% of
  FY2008 certified positions; the rest use a state average of area values or
  are missing [04].
- **OES vintage:** May 2007 data for receipts before July 2009, May 2008
  after. This follows the July wage-year convention, which is unverified
  locally.
- **Effective floor:** the maximum of the proxy and the applicable minimum
  wage. The minimum wage binds in 20% of exposure-weighted counties [05].
- **Exposure:** log(2010 FLS AEWR / effective 2009 floor).

| Statistic | Value | Script |
|---|---|---|
| Counties with FY2008 single-county certifications | 598 | [05] |
| Exposure, $ (FLS 2010 − effective floor), FY2008-weighted mean (SD) | 2.08 (0.71) | [05] |
| Log exposure, mean (SD) | 0.241 (0.089) | [05] |
| Share of exposure variance within AEWR region / within state | 0.60 / 0.32 | [05] |
| Offered wage on effective floor, state + month FE: 1987 rule | 0.011 (0.019) | [05] |
| same: 2008 rule | 0.293 (0.059) | [05] |
| same: 2010 rule | 0.033 (0.030) | [05] |
| Interaction, 2008 rule × floor (pooled) | 0.240 (0.060) | [05] |
| 2008-rule cases within 10¢ of the effective floor proxy | 18.7% | [05] |
| States: corr(Table 1 post-Jan-19 wage, proxy) / corr(Table 1 drop, proxy − 2008 FLS) | 0.44 / 0.22 (42 states) | [04] |
| Table 1 state wage drop, SD / within-region SD | $1.02 / $0.82 | [04] |

Reading:
- **The proxy has real signal.** Offered wages track it only under the 2008
  rule.
- **The proxy is noisy or overstates the cut.** The slope of 0.29 is far
  below 1, and the dollar exposure ($2.08) is well above the observed drop
  (about 12%). Three reasons:
  - The OES vintage and skill level are approximated.
  - Prevailing wages and other floors bind for many cases.
  - Many employers did not cut to the new floor.
- **Use it as a reduced-form exposure, not a wage change.** Treat the proxy
  as an exposure measure with a first stage, not as the wage change itself.

## 4. County response: descriptive check

PPML of county certified workers and applications on year × standardized
exposure, with county and region-by-year FE, clustered by state, reference
FY2008 [06]:

| Year | Workers | Applications |
|---|---|---|
| 2009 | 0.083 (0.020) | 0.072 (0.051) |
| 2010 | 0.058 (0.039) | 0.064 (0.033) |
| 2011 | 0.132 (0.065) | 0.066 (0.039) |
| 2012 | 0.210 (0.090) | 0.085 (0.045) |
| 2013 | 0.278 (0.122) | 0.127 (0.050) |

The 2009 coefficient has the sign a temporary wage cut predicts. But the
relative growth continues and increases after FLS floors were restored in
FY2011. That pattern fits differential trends: low-local-wage counties within
a region were H-2A growth areas. It does not fit a response to the reversal.
With a single pre-year, the trend cannot be separated from the effect.

## 5. Design options

| Option | Variation | Main threats | Verdict |
|---|---|---|---|
| Filing-date RD at 2009-01-17 / 2010-03-15 | Regime by filing date | Filing timing chosen (spikes above); other rule provisions change too | Not credible for use; fine for the wage first stage |
| County continuous-exposure DiD, FY2008–13 | Local floor proxy × regime | One pre-year; trends correlated with exposure; regime mixing in FY2009–10; proxy error | Not credible now; needs FY2006–07 data and verified FLC schedules |
| Employer cohort (FY2008 incumbents) × exposure, FY2008–12 | Same exposure, fixed incumbent geography | Same pre-trend issue; linkage attrition (already flagged in `wage_reforms.md`); temporary, anticipated cut | Best candidate for a short-run response, if pre-trends can be added |
| Case-level pass-through | Offered wage vs local floor, by regime | Proxy measurement | Feasible now; documents bite and incidence |

## Open blockers

1. **The verified FLC Level I–IV wage files for July 2008–June 2010.** These
   would replace the proxy and settle which OES vintage applied.
2. **FY2006–07 H-2A disclosure data.** These would give pre-trends for any
   exposure DiD.
3. **The injunction order date and the proposed-suspension date**, from the
   M.D.N.C. docket and the Federal Register. These would fix the
   uncertain-regime window. The ~12% of post-June-29 cases at FLS rates needs
   explaining, either as partial implementation or as employer hedging.
4. **The text of the 2010 transition FAQ.** Only a summary is stored locally.
