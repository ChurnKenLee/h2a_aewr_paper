# AEWR bite designs: county exposure (1a, 1b) and employer bindingness (4)

Scratch diagnostics, not pipeline products. Scripts are in the session
scratchpad under `bite/`: `county_bite.R`, `bind_check.R`, `employer_bite.R`,
`employer_robust.R`. Real AEWR is nominal AEWR / PPI (2012 dollars).

## Designs

- **1(b)**: y_ct = θ · E_c · A_rt + county FE + region×year FE. Uses the
  national rise in the AEWR, so it rests on parallel trends across exposure.
- **1(a)**: adds E_c × year FE. Only region-specific AEWR deviations identify θ.
- **4**: the same two designs at the employer level, with E replaced by
  baseline bindingness (whether the employer paid the AEWR in 2011–2012).
- **5**: the time-varying bite, AEWR minus the current local wage, as in the
  retained DiD (section 5; added 2026-11-12).

## 1. County exposure from Census pay: not viable

Exposure E_c = minus log 2007 Census of Agriculture hired-labor expense per
hired worker, centered within AEWR region and scaled to unit SD (2,942 counties
with at least 20 hired workers; within-region SD of log pay 0.54).

**The exposure does not measure bite.**

- Among 1,425 counties using H-2A in 2011–2013, E does not predict the share of
  hourly H-2A hours offered at the AEWR: coefficient −0.006 (SE 0.013), within
  region.
- Adjusting pay for the share of workers employed under 150 days leaves a
  measure uncorrelated with the original (correlation 0.005). Pay per worker
  mostly measures employment duration, not wage levels.

**The AEWR binds almost everywhere.** 94% of 2011–2013 hourly H-2A hours are
offered at the AEWR (hours-weighted); the median county is at 100%. No local
wage measure predicts which counties are less bound. Within-region R² of the
county at-AEWR share on each measure:

| Measure | Within R² |
|---|---|
| OEWS Big-Six wage / AEWR | 0.0003 |
| CZ p25 wage | 0.0015 |
| p10 wage / AEWR | 0.0008 |
| State minimum wage / AEWR | 0.0000 |

**Estimates** (2011–2024, 41,118 county-years, SE clustered by state):

| Outcome (per 2011 farm worker) | 1(b) | 1(a) | 1(a) MDE (2.8 SE) | Outcome mean |
|---|---|---|---|---|
| Certified workers | −0.0011 (0.0021) | −0.0061 (0.0170) | 0.048 | 0.095 |
| Certified hours | −1.9 (1.3) | 3.8 (12.5) | 35 | 90 |
| Applications | −0.00018 (0.00005) | 0.00098 (0.00049) | 0.0014 | 0.0059 |
| Linked businesses | −0.00054 (0.00012) | 0.0015 (0.0016) | 0.0045 | 0.013 |

- **1(b) traces a trend, not the AEWR.** The applications and businesses
  coefficients are significant. But the E × year event-study coefficients
  for applications rise from 2008 into 2010 and then decline smoothly through
  2024, with no break when the window starts in 2011. With the AEWR rising almost linearly, this cannot be told
  apart from a differential trend. The duration-adjusted exposure gives nulls
  for every outcome.
- **1(a) has no power.** Its SEs are 8–14 times those of 1(b). The
  certified-worker MDE is half the outcome mean.

## 4. Employer bindingness: one promising result

**Cohort.**

- 6,684 v2 balanced worksite businesses with hourly certified hours in start
  years 2011–2012 and a single baseline AEWR region.
- Bound = at least half of baseline hourly hours offered at the AEWR, using the
  b01 rule: within $0.05 of the start-year or prior-year state AEWR.
- 5,977 bound, 707 unbound. Unbound employers hold 5.5% of baseline hours and
  offer a median 1.04 × AEWR (90th percentile 1.33).

**Where the unbound employers are.** They are concentrated by geography and
by type:

- **States:** North Dakota 258, Texas 139, Kansas 82, California 29.
- **Unbound share by region:** California 56%, Northern Plains 56%, Southern
  Plains 37%. In 11 of 17 regions, 8% or fewer employers are unbound.
- **Regions with both groups:** only the Delta, Southern Plains, California and
  Northern Plains have at least 20 employers in each group.

**Outcomes.** Employer-year participation (any certified workers) and log(1 +
certified workers), with zeros for nonparticipation, from 2008 to 2024.

**Estimates.** The coefficient is on bound × real regional AEWR, with employer
FE and region×year FE. SEs are clustered by baseline state.

| Model | Participation | log(1 + workers) |
|---|---|---|
| 1(b)-analog, 2011–2024 | −0.0001 (0.0021) | −0.015 (0.007) |
| 1(a)-analog, 2011–2024 | −0.012 (0.011) | −0.052 (0.030) |
| 1(b)-analog, 2011–2019 | −0.0017 (0.0030) | −0.022 (0.009) |
| **1(a)-analog, 2011–2019** | **−0.055 (0.016)** | **−0.130 (0.051)** |

**Robustness of the 1(a)-analog, 2011–2019:**

- Leave one region out (dropping the Delta, Southern Plains, California or
  Northern Plains in turn): participation −0.045 to −0.059; log workers −0.088
  to −0.161.
- The four mixed regions alone: −0.078 (0.022) and −0.215 (0.070).
- Restricted Webb wild cluster bootstrap over 17 regions: p = 0.009
  (participation) and p = 0.048 (log workers).

**Reading.** When a region's AEWR rises $1 (2012 dollars) above its trend
relative to other regions, bound employers become 5.5 percentage points less
likely to participate than unbound employers in the same region. Baseline
participation in 2013–2019 is roughly 0.27–0.50. Their certified workers fall
about 13% relative to unbound employers.

**Caveats, in order of importance:**

1. **Unbound employers are a different kind of business.** They are mostly
   Plains employers, plausibly custom harvest and equipment crews. Regional
   shocks such as grain prices can hit them differently from bound employers,
   and that would bias the 1(a)-analog. Job titles exist in the case data and
   could separate employer types; that is the next test.
2. **The result weakens through 2024.** The 2011–2024 estimate is a fifth the
   size and insignificant. From 2020, member businesses named on association
   and labor-contractor filings appear (employer counts in the case data go
   from about 10,000 in 2019 to 36,000 in 2020), which breaks comparability. That is a reason to stop
   at 2019, not a proof that the later years are uninformative.
3. **The event study shows pre-period differences.** Participation is higher
   for bound employers before 2012 (0.07–0.09 relative to 2012), because the
   cohort is selected on 2011–2012 participation. After 2013 the gap is flat at
   about 0.045. Bound × year FE absorb this level path in the 1(a)-analog.
4. **Inference rests on few contributing regions.** The bootstrap is over 17
   regions, but most of the identifying contrast comes from four.

**Next checks for design 4:**

- Classify baseline job titles into harvest/equipment versus field work, and
  estimate within type, or with type × region × year FE.
- A placebo using next year's regional AEWR deviation.
- An alternative cohort defined in 2015–2016, to test stability before the
  2020 break.
- Estimates on the 2020–2024 cohort under the new linkage regime.

## 4b. Follow-up checks on design 4

Scripts: `employer_checks.R`, `employer_windows.R`, `employer_persistence.R`.
Employer type is the hours-weighted majority of baseline job titles, assigned
by regex:

- **equipment:** equipment, machine, operator, combine, truck, driver, tractor,
  custom, mechanic;
- **livestock:** animal, livestock, ranch, dairy, herd and similar;
- **crop:** everything else.

The occupation code is empty for every 2011–2012 baseline case, so titles are
the only option.

**The original estimate survives the type and placebo checks.** These are
1(a)-analog estimates for participation and log(1 + workers), with SE clustered
by state:

| Check | Participation | log(1 + workers) |
|---|---|---|
| Original 2011–12 cohort, 2011–19 | −0.055 (0.016) | −0.130 (0.051) |
| Crop-titled employers only (242 unbound, 5,304 bound) | −0.079 (0.026) | −0.200 (0.081) |
| Equipment-titled only (370 unbound, 359 bound) | −0.038 (0.018) | −0.049 (0.074) |
| Type × region × year FE and bound × type × year FE | −0.056 (0.016) | −0.134 (0.054) |
| Adds bound × next-year AEWR: current term | −0.047 (0.019) | −0.109 (0.047) |
| Adds bound × next-year AEWR: lead term | −0.015 (0.020) | −0.037 (0.045) |

**The estimate does not replicate in other cohorts:**

| Cohort and window | Participation | log(1 + workers) |
|---|---|---|
| 2011–12 cohort, 2014–19 only | −0.042 (0.015) | −0.115 (0.035) |
| 2014–15 cohort, 2014–19 | +0.022 (0.021) | −0.024 (0.053) |
| 2020–21 cohort, 2020–24 | +0.091 (0.099) | −0.029 (0.056) |

Over the same 2014–19 years, the 2011–12 cohort shows the effect and the
2014–15 cohort does not. The result is a property of the 2011–12
classification, not of those years' AEWR variation.

**Unbound status is mostly transient.**

- 3,108 of the 6,684 cohort employers are also in the 2014–15 cohort.
- Of the 256 that were unbound in 2011–12, 162 (63%) are bound in 2014–15.
  Among crop-titled employers the figure is 90 of 117.
- Of 2,852 bound employers, 45 (1.6%) become unbound.
- 2011–12 unbound employers are less likely to remain in the program by
  2014–15 (36% against 48% for bound).

Paying above the AEWR is not a stable employer trait. It is closer to a
temporary state, or to measurement error in matching offered wages to the
applicable AEWR. The 2011–12 contrast therefore compares one particular,
mostly transient group with everyone else, and later cohorts do not reproduce
it.

**Verdict on design 4:** not robust enough to support a causal estimate. It
passes the leave-one-region-out, wild bootstrap, employer-type and lead
placebo checks. It fails replication across cohorts, which a real bindingness
mechanism should pass.

## 1b. County exposure from the AEWR-relative-to-local-wage bite

Script: `county_bite_acs.R`. This repeats designs 1(a) and 1(b) with the
repository's own bite concept instead of Census pay. The bite is frozen over
2008 and 2010 (the 2009 AEWR is missing):

- **gap:** real AEWR − max(ACS commuting-zone p25 real hourly wage, real
  applicable agricultural minimum wage).
- **fraction affected:** the share of the five ACS quantile point masses
  (p10–p90) below the AEWR after the minimum-wage floor, as in the
  Mundlak–Chamberlain branch.

Both are centered within AEWR region and scaled to unit SD.

The earlier check in section 1 asked whether these measures predict the share
of H-2A hours paid exactly the AEWR. That was the wrong test for this bite,
which measures how expensive H-2A labor is relative to local domestic labor,
not whether the AEWR binds on H-2A offers.

**Variation.** 3,108 counties in 608 commuting zones.

- The baseline gap averages $1.14, with SD $1.31 and within-region SD $1.03.
  62% of its variance is within region.
- Fraction affected averages 0.36 but has a within-region SD of only 0.06. The
  five-point approximation is coarse.
- The two measures correlate 0.75.

**Relevance in levels, 2011, within region:**

- A 1 SD larger gap goes with a 3.5-percentage-point lower probability of any
  H-2A use (SE 1.5).
- It has no relation to certified workers per farm worker: 0.0016 (SE 0.0039).

**Estimates** (2011–2024, 42,574 county-years, SE clustered by state;
coefficient on E × real AEWR):

| Outcome | 1(b), gap | 1(a), gap | 1(b) MDE | 1(a) MDE | Mean |
|---|---|---|---|---|---|
| Certified workers / 2011 farm workers | 0.0025 (0.0023) | −0.024 (0.025) | 0.0066 | 0.069 | 0.095 |
| Certified hours / 2011 farm workers | 1.8 (1.8) | −14.5 (16.9) | 4.9 | 47 | 90 |
| Applications / 2011 farm workers | −0.00001 (0.00004) | −0.00027 (0.00037) | 0.00011 | 0.0010 | 0.0059 |
| Linked businesses / 2011 farm workers | 0.00003 (0.00013) | −0.0013 (0.0015) | 0.00036 | 0.0042 | 0.013 |

Fraction-affected exposure gives the same pattern: all eight coefficients
have |t| ≤ 1.02.

**Reading:**

- **1(b) is a precise null.** For each $1 real AEWR increase, a county whose
  baseline gap is 1 SD (about $1) larger shows no differential change in H-2A
  use. The 95% interval for certified workers per farm worker is about
  [−0.002, +0.007], which rules out differential responses larger than about
  7% of the mean. The event study shows no pre-trend in 2008–2010 and a slow,
  insignificant upward drift after 2011.
- **1(a) remains uninformative.** Its MDEs are as large as the outcome means.
- **What 1(b) identifies.** It identifies how the effect differs with bite,
  not the average AEWR effect. It is evidence against a large AEWR effect only
  if the effect should be larger where the AEWR sits further above local
  wages. That is the standard minimum-wage-bite prediction, but the paper
  would need to argue it for H-2A.

## 5. Time-varying bite (AEWR minus current local wage): does not work either

Added 2026-11-12. Script: `explorations/bite/timevarying_bite_check.R`. It
reads the retained DiD panel, prints to stdout and writes nothing. No retained
table or specification changed.

**Question.** Sections 1 and 1b freeze the local wage. The economically
relevant bite is the current gap, so the wage should vary:

B_{c,t−1} = real AEWR_{r,t−1} − real CZ p25 wage_{c,t−1} (`aewr_cz_p25_l1`).

This is the regressor of the retained DiD. The check uses that branch's sample
and controls (`table_1_main_results.tex`, column 2): 32,116 county-years,
2,294 counties, 2008–2022, county and year FE, SEs clustered by
CZ × AEWR region. The outcome is certified positions per 2011 farm worker
(sample mean 0.041). The check reproduces the retained coefficients (main
0.0036, post interaction −0.0071).

**Verdict: the time-varying bite does not identify an AEWR effect either.**

| Specification | Coefficient (SE) |
|---|---|
| Bite, one slope for all years | −0.0023 (0.0052) |
| Bite slope after 2011 (main + post interaction) | −0.0035 (0.0051) |
| Bite, county + region×year FE (only the wage identifies) | 0.0049 (0.0085) |
| Bite, county + state×year FE | 0.0051 (0.0099) |
| AEWR entered separately | −0.032 (0.013); region-clustered SE 0.022 |
| CZ p25 wage entered separately | −0.012 (0.009); region-clustered SE 0.010 |

- **The bite has no detectable level relationship with H-2A use.** The
  retained −0.0071 is a change in slope at 2011. The post-2011 slope itself
  is −0.0035 (0.0051).
- **The wage component has the wrong sign.** A bite model requires the wage
  coefficient to equal minus the AEWR coefficient, about +0.032. It is
  −0.012. The restriction b_AEWR + b_wage = 0 gives −0.044 (SE 0.020,
  p = 0.025) with the retained clustering, and SE 0.031 (p = 0.18) with 17
  region clusters. So the rejection depends on the clustering level.
- **The bite coefficient is near zero because two same-signed components
  cancel.** H-2A use falls when the AEWR rises and does not rise when the
  local wage rises.
- **The retained post-2011 interaction comes from the AEWR component.** With
  the components separated, AEWR × post is −0.025 (0.009) and wage × post is
  −0.006 (0.006). With region×year FE, where only the wage identifies, the
  bite × post interaction is +0.011 (0.007).
- **The AEWR component alone is the plain panel design.** It varies by
  region-year across 17 regions. Its coefficient implies an elasticity of
  roughly −8, as implausible as the plain-panel and panel-IV magnitudes, and
  it is not significant with region clustering.

**Where the variation comes from.** Within county and year, the bite's
variance is 0.31. The wage contributes 0.27 and the AEWR 0.16, and the two
are positively correlated (0.31), which lowers the bite's variance. After
region×year FE the bite's SD is $0.46, all from the wage.

**Reading.** The weak link is the wage measure, not whether it varies. The CZ
p25 covers all workers, not farm workers, and takes a median of 313 distinct
values per year across 571 CZs. Its movements may track local growth that
also reduces farming, which would explain the negative sign. That reading is
not tested here.

**Not run:** a time-varying bite built on a farm wage (county QCEW or QWI
crop pay). It measures the right wage, but H-2A demand shocks move farm wages
directly, so it would need its own exogeneity argument.
