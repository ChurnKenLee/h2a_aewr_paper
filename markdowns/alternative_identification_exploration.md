# Alternative identification for the AEWR effect on H-2A use: exploration report

This report executes `markdowns/alternative_identification_agent_plan.md`.
Scripts are in `explorations/identification/` (see its README). Numerical
outputs are in the scratch directory
`/tmp/claude-1000/-home-ken-Dropbox-projects-H-2A-Paper/1be305e6-0442-48c5-93f4-7023961621fb/scratchpad/identification/`
(called `$S` below). Every number here comes from a CSV written by the script
named next to it.

**Update, 2026-11-02 (after your decisions).**

- You approved reading FY2026, amending Track 1, acquiring the overtime
  schedule and fixing the minimum-wage coverage coding. You also confirmed
  the rule's legal status.
- **Track 1** was amended before any FY2026 read (new registry and hash), then
  estimated (§3).
- **Track 4** was acquired and estimated (§6).
- **The coverage producer** was fixed in code and Track 2 rerun on the
  corrected coding (§4).
- **Nothing was regenerated in `data/`, `outputs/` or the shared panel.** No
  retained table changed. The only repository code changes are the two
  minimum-wage producers and the new tracked coverage table.
## Verdict

**There is still no design that pins down the average AEWR effect. There is
now a credible design that bounds it from below.**

- **The 2025 statewide rule (Track 1, estimated).**
  - **First stage:** the rule cut the statutory skill-I rate in every sample
    county (mean −$4.09), and offered wages fell. Within former regions,
    offered wages fall $0.66 per statutory dollar of cut (jackknife SE
    0.095).
  - **Quantity response:** none detectable one season later. Positions
    elasticity 0.40 [−1.06, 1.86]; hours 0.21 [−0.85, 1.27]; size-weighted
    PPML −0.12 [−0.77, 0.53].
  - **Placebo** (FY2024 vs FY2025): clean.
  - **Power:** the realized elasticity MDE is 2.0 on the primary outcome
    (0.9 for PPML), so the 1.0 gate fails on the declared primary.
  - **Takeaway:** it rules out highly elastic demand (ε ≲ −1). It does not
    distinguish conventional elasticities (−0.15 to −0.75) from zero.
- **Region-year designs are capped.** Only 1.4–5.7% of residual outcome
  variance lies at the region-year level. PPML and a baseline-employer panel
  reach elasticity MDEs of 1.5–2.1. The FLS revision instrument has no
  variation.
- **State minimum wages (Track 2, corrected coverage coding):** a
  well-powered, pre-trend-clean null for the domestic floor.
  - **No independent AEWR bound.** Reading it as an AEWR effect runs through
    the same "only the AEWR-minus-domestic-wage gap matters" model as the
    bite design, so it is not independent of the bite.
  - **Unestimated pass-through.** It needs the minimum wage to raise the
    competing domestic wage one-for-one (λ = 1), which was never estimated.
  - **Bias toward zero.** It ignores scale effects, which bias the
    translation toward zero (§4).
- **Multi-region employers (Track 3):** a mechanism check at best.
- **Farm overtime laws (Track 4, acquired and estimated):** not credible.
  The lead fails, OLS and PPML disagree in sign, and randomization inference
  does not reject. Keep the schedule as a sourced input for the welfare
  model's hours margin.
- **Where this leaves the evidence.** One credible design, Track 1, rules
  out elasticities below about −1 one season after the 2025 cut.
  - **The gap-model designs** (the bite conversion and the Track 2
    translation) are consistent with that. They rest on one shared,
    untestable model of the domestic alternative wage, so they are not
    independent confirmation.
  - **Nothing yet separates** conventional elasticities (−0.15 to −0.75)
    from zero.
  - **Not adequate for the paper.** You judged this inadequate on
    2026-11-02. The broader search for identification strategies is planned
    in `markdowns/identification_search_agent_plan.md`.

## 1. One unit for every design

`01_common_unit.R` refits the plain panel, panel IV and bite 1(a)/1(b) models
on their original samples. It reproduces the retained numbers: OLS −0.04427
and IV −0.08931 on certified positions per 2011 farm worker. It then converts
every headline estimate to an elasticity at estimation-sample means,
β × mean(price) / mean(outcome). Output: `$S/common_unit_all.csv` and
`$S/common_unit_headline.csv`.

On the retained ratio outcome the mean real AEWR is $12.92 and the mean
outcome is 0.091. An elasticity of 1 is therefore a slope of about 0.007
positions per farm worker per real dollar. The plain-panel standard error,
0.039, is more than five times that.

| Design (outcome) | Estimand | β (SE) | Wild p | Elasticity [95% CI] | Elasticity MDE | MDE % mean |
| --- | --- | --- | --- | --- | --- | --- |
| Plain panel OLS (positions) | average effect | −0.0443 (0.0387) | 0.36 | −6.3 [−18.0, 5.4] | 15.4 | 119 |
| Plain panel OLS (hours) | average effect | −31.9 (24.8) | 0.22 | −4.8 [−12.8, 3.2] | 10.6 | 82 |
| Panel IV col. 4 (positions) | average effect, IV | −0.0893 (0.0628) | 0.26 | −12.7, AR [−31.8, 9.0] | 25.0 | 193 |
| Panel IV col. 4 (hours) | average effect, IV | −65.5 (39.5) | 0.13 | −9.9, AR [−22.8, 4.6] | 16.8 | 130 |
| Bite 1(b) (positions) | differential slope per 1 SD frozen gap | 0.0015 (0.0012) | 0.24 | 0.22 [−0.14, 0.57] | 0.47 | 3.6 |
| Bite 1(b) (hours) | same | 1.35 (1.12) | 0.29 | 0.20 [−0.15, 0.56] | 0.47 | 3.7 |
| Bite 1(b) → average, positions | average under β_c ∝ frozen gap | 0.0018 | 0.24 | 0.26 [−0.17, 0.68] | 0.56 | 4.4 |
| Bite 1(a) (positions) | differential, exposure × year FE | −0.0117 (0.0157) | 0.76 | −1.7 [−6.4, 3.1] | 6.2 | 48 |
| Border v6 ring, baseline (positions) | average border effect, pair CRV1 | −0.0090 (0.0077) | — | −1.8 [−4.9, 1.3] | 4.2 | 32 |
| Border, screened, bite interaction (positions) | differential per 1 SD pair exposure | 0.0374 (0.0141) | — | 8.5 [1.9, 15.1] | 9.0 | 68 |
| Employer design 4, 1(a) 2011–19 (log1p workers) | bound vs unbound; fails replication | −0.130 (0.051) | — | −1.6 | 1.8 | — |
| Statewide border, power only (positions, FY) | design-stage MDE, worst shared DGP | — | — | — | 3.7–6.8 | 24–44 |

Notes:

- The bite rows are differential effects per standard deviation of exposure,
  not average effects. The "→ average" row imposes the strong restriction of
  bite report §6.
- The border rows use the bite cross-design scratch outputs
  (`/tmp/h2a-bite-cross-design/border_estimates.csv`). Their wild p-values are
  in that directory, not recomputed here.
- Employer design 4 is converted from the bite report §4 numbers. Its
  participation elasticity uses the midpoint (0.385) of the reported baseline
  participation range, so treat it as approximate.
- Mundlak–Chamberlain is omitted. Its fraction-affected exposure mixes
  nominal policy wages with real quantiles (bite cross-design report §7), so
  it cannot be put on this scale without first fixing that unit mismatch.
- The 2009–10 reversal and the herder phase-in produced offered-wage
  responses but no county quantity elasticity, so they have no row.

**Gates.** The proposed gate (elasticity MDE ≤ 1.0) is passed only by the
bite 1(b) interaction, which does not identify the average effect. The earlier
gate (MDE < 25% of the mean per real dollar) is passed by the same rows. Every
average-effect design fails both, by a factor of 3 to 25.

## 2. Phase 0: signal per region-year, and support

### 2a. The FLS revision instrument is dead

`00_fls_revision.R` (`$S/fls_revision_summary.csv`, `$S/fls_aewr_match_summary.csv`):

- The parser stores, for each annual release, the older year as "revised" and
  the newer as "preliminary". Matching preliminary(t) from release t to
  revised(t) from release t+1 compares **different releases in all 391
  region-years** (2001–2023; `same_release` share 0).
- The published AEWR for year t+1 equals the preliminary field-and-livestock
  wage for year t in **391 of 391** region-years (maximum difference $0).
- Preliminary equals revised in 96.2% of region-years (2001–2023) and 99.5%
  over 2010–2022. The within-year SD of the revision over 2010–2022 is
  **$0.00065**. The 15 nonzero revisions are $0.01–0.02, mostly in 2005.

There is no variation to instrument with. The design is dead.

### 2b. More signal per region-year

`02_precision.R` (`$S/precision_estimates.csv`, `$S/precision_region_year_share.csv`).
The treatment is the real regional AEWR throughout, with county (or employer, or
region) and year FE. Only the outcome, weighting, aggregation and estimator
change. County specifications use the panel-IV BEA-control sample, 2011–2023,
36,239 county-years and 17 region clusters. Elasticities for PPML, the
aggregate and the employer panels are β × mean AEWR (log or semi-elasticity).

| Specification | Outcome | Estimate (SE) | Wild p | Elasticity [95% CI] | Elasticity MDE | Region-year share of residual variance |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Ratio, equal weights, OLS | positions | −0.0443 (0.0387) | 0.36 | −6.30 [−17.97, 5.38] | 15.4 | 0.051 |
| 1. Ratio, equal weights, IV | positions | −0.0893 (0.0628) | 0.26 | −12.70 [−31.63, 6.22] | 25.0 | |
| 2. PPML counts, OLS | positions | 0.0774 (0.0580) | — | 1.00 [−0.59, 2.59] | 2.10 | 0.014 |
| 2. PPML control-function IV | positions | −0.0701 (0.2102)* | — | −0.91 [−6.68, 4.86] | 7.62 | |
| 3. Ratio, 2011 farm-emp weights, OLS | positions | 0.0056 (0.0195) | — | 0.91 [−5.79, 7.61] | 8.85 | 0.057 |
| 3. Same, IV | positions | −0.0253 (0.0432) | — | −4.11 [−18.99, 10.77] | 19.7 | |
| 4. Region-year log total, OLS | positions | 0.148 (0.085) | 0.20 | 1.91 [−0.41, 4.22] | 3.06 | 1 (by construction) |
| 4. Same, IV (first-stage F 18.9) | positions | 0.343 (0.212) | — | 4.42 [−1.38, 10.23] | 7.67 | |
| 5. Employer panel 2011–19, log1p | positions | 0.0009 (0.0436) | 0.99 | 0.01 [−1.12, 1.14] | 1.49 | 0.041 |
| 5. Employer panel 2020–25, log1p | positions | −0.0388 (0.0539) | 0.53 | −0.56 [−2.19, 1.08] | 2.16 | 0.025 |
| 1. Ratio, OLS | hours | −31.9 (24.8) | 0.22 | −4.85 [−12.85, 3.15] | 10.6 | 0.037 |
| 2. PPML counts, OLS | hours | 0.110 (0.055) | — | 1.42 [−0.09, 2.92] | 1.99 | 0.014 |
| 2. PPML control-function IV | hours | 0.029 (0.194)* | — | 0.37 [−4.96, 5.70] | 7.04 | |
| 4. Region-year log total, OLS | hours | 0.179 (0.095) | 0.20 | 2.31 [−0.28, 4.90] | 3.42 | 1 |
| 5. Employer panel 2011–19, log1p | hours | −0.091 (0.146) | 0.55 | −1.11 [−4.88, 2.66] | 4.98 | 0.030 |

\* Region pairs bootstrap, 199 draws, both stages refit (naive SEs 0.205 and
0.185). Employer panels: 7,569 employers present in 2011–12 and 57,745 present
in 2020–21, each with a single baseline region and zeros kept. Their
treatment is the baseline region's AEWR.

What this shows:

- **The region-year share of residual variance is 1.4–5.7%** on every
  county and employer outcome. That is the quantity that limits every
  region-year design, and no change of outcome or estimator raises it.
- **PPML is the largest gain.** It cuts the elasticity MDE from 15.4 to 2.1
  (positions) and from 10.6 to 2.0 (hours), because it weights by size and
  the ratio outcome is dominated by noise in small counties. The 2011–19
  employer panel reaches 1.49. **None reaches 1.0.**
- **The IV versions stay imprecise** (MDE 7–25). The control-function PPML-IV
  is about as imprecise as the equal-weight IV once the first stage is
  bootstrapped.
- **The non-IV point estimates are positive** wherever precision improves
  (PPML +1.0 and +1.4, aggregate +1.9 and +2.3). That is the sign expected if
  regional AEWR deviations track regional demand, because the AEWR is built
  from regional farm wages. These are not causal estimates without the IV
  exclusion.
- **Precision gains do not touch exclusion.** PPML, weighting and aggregation
  change only the noise. The region-year AEWR deviation must still be
  exogenous, or the donor exclusion must hold.
- **Weighting changes the estimand.** PPML and farm-employment weights
  estimate a size-weighted response (closer to the aggregate response the
  welfare model needs). The equal-weight ratio estimates the average county's
  response. They are not interchangeable.

### 2c. Treatment support for the new tracks

| Track | Identifying variation after its own FE | Support |
| --- | --- | --- |
| 1. 2025 statewide rule | FY2026 statewide U.S. skill-I rate within former region: SD $1.09 across states, $0.80 county-weighted; largest within-region range $3.81 | 46 states in 15 multi-state former regions; 2,921 counties. California (17) and Florida (11) are single-state regions and drop out; Hawaii, Alaska and DC are excluded. Adequate. (`03_track1_support.R`, `$S/track1_support.csv`) |
| 2. State minimum wages | Residual SD of the real applicable agricultural floor after county + region × year FE: $0.59 (general minimum $0.63) | 275 of 816 state-years (2009–2025) change the nominal agricultural floor, 162 by at least 5%. Frozen CZ p10 is within $1 of the floor in 18.4% of counties, p25 in 10.4%. Adequate. (`05_track2_minwage.R`, `$S/track2_support_*.csv`) |
| 3. Multi-region employers | Employer-years placing workers in ≥ 2 AEWR regions | 2011–19: 160 employers a year on average, 16.8% of positions and 12.6% of hours, rising from 51 (2.9%) in 2011 to 224 (27.8%) in 2019. 2020–25: 294 a year, 7.3% of positions, after the linkage break. Adequate. (`06_track3_multiregion.R`, `$S/track3_support_*.csv`) |

For comparison, the region-year designs use about 220–240 region-year
deviations with an SD of $0.40–0.46. Track 1's within-region dispersion is two
to three times larger but appears in one post year (two after FY2027).

## 3. Track 1: the 2025 statewide rule, all counties

### Specification (declared)

The registry is `explorations/identification/track1_registry.json`, frozen
2026-11-02 before any FY2026 data were read. **SHA-256
`bbffa053953873aa8a36c04a5af71c012fd15460b2cd0ff84bd378f6e65aafd8`.**

- **Unit and sample:** county × federal fiscal year, FY2012–FY2026. All
  counties with positive 2011 farm employment in multi-state former AEWR
  regions (2,921 counties, 46 states).
- **Treatment:** the rate in effect on April 1 of each FY. Before FY2026 this
  is the regional FLS reference; in FY2026 it is the statewide U.S.-worker
  skill-I rate, in nominal dollars. Sensitivities are the H-2A skill-I cash
  rate after the housing adjustment, U.S. skill II, and a cohort-aligned
  version.
- **Fixed effects:** county and former region × FY. Only post-2025 dispersion
  within a former region identifies the slope. The national level change is
  absorbed and **not identified**. The mean dose is −$3.96: the statewide
  skill-I rate is well below the FY2025 regional rate almost everywhere.
- **Outcomes:** certified positions and certified hours per fixed 2011 farm
  worker, counting only cases decided October–June (Q1–Q3) in every FY. The
  full-year pre-period outcome is a check. PPML counts and the offered-wage
  first stage are also declared.
- **Construction:** `04_track1_power.R` rebuilds fiscal-year outcomes with the
  b01 fiscal-year logic: allocated worksite rows, equal county split, and
  period-date hours. It reproduces the producer's full-year county positions
  exactly in every FY2012–FY2025, with county correlation 1 and maximum county
  difference below 1e−12. Hours match within 0.003%; b01 imputes missing
  weekly hours with a pooled mean that may include later rows
  (`$S/track1_outcome_validation.csv`). Decision dates exist for 100% of
  FY2012–2025 cases. Q1–Q3 decisions carry 79–82% of positions and 83–87% of
  hours each year (`$S/track1_q13_share.csv`,
  `$S/track1_decision_date_coverage.csv`).
- **Inference:** CRV1 by state as primary; a state jackknife and
  former-region CRV1 as alternatives; a restricted Webb wild bootstrap over
  states; and randomization inference permuting doses across states within a
  former region.

### Design-stage power

`04_track1_power.R` (`$S/track1_power_*.csv`) uses the plan's machinery: the
border branch's four error processes, seed 201103, 999 repetitions, AR(1) 0.5,
and the upper-Wilson size gate at 0.075. It runs on all 2,921 counties and the
actual FY2026 (and FY2027) statewide rates. The scale comes from FY2020–FY2025
residuals after county and former-region × FY FE only. The border "region"
shock becomes a state shock and the "pair" shock a former-region shock, which
the FE absorb. Calibration means: positions 0.113 per 2011 farm worker, mean
nominal rate $13.15, so an elasticity of 1 is β = 0.0086.

**Leverage is concentrated.** The effective number of states is 11.1 of 46.
The largest single state carries 24% of the treatment variation.

U.S. skill-I treatment, FY2026 only, upper MDE across signs:

| Error process | State CRV1: null rejection / MDE (elasticity) | State jackknife CV3: null rejection / MDE (elasticity) | Former-region CRV1: null rejection / MDE (elasticity) |
| --- | --- | --- | --- |
| County AR(1) only | 0.162 ✗ / 0.0085 (0.99) | 0.026 ✓ / 0.0130 (1.51) | 0.101 ✗ / 0.0103 (1.20) |
| 50% state + 50% county | 0.169 ✗ / 0.055 (6.4) | 0.027 ✓ / 0.085 (9.8) | 0.086 ✗ / 0.067 (7.8) |
| State + former region + county | 0.173 ✗ / 0.042 (4.9) | 0.030 ✓ / 0.066 (7.7) | 0.092 ✗ / 0.053 (6.1) |
| Spatial state + region + county | 0.173 ✗ / 0.061 (7.1) | 0.024 ✓ / > grid (> 10) | 0.093 ✗ / 0.075 (8.6) |

- **Hours** look the same. Jackknife MDE elasticities are 1.30 (county only)
  and 6.6–9.4 (shared).
- **Adding FY2027** helps only a little: positions jackknife MDE 0.0104
  (1.21) with county-only errors, and 6.0–8.6 with shared shocks.
- **Calibrating on FY2012–FY2025 makes it worse.** County-only MDE rises to
  3.8, and shared-process MDEs leave the grid. Pre-2020 residuals are larger
  relative to the outcome.
- **The first stage** (offered wage, 2,260 counties with offers in every
  calibration year) has a jackknife MDE of 0.07 (county only) to 0.43
  dollars per statutory dollar (shared). It is powered to detect full
  pass-through but not the registry's 0.25 under shared shocks.
- **Decision table** (`$S/track1_power_decision.csv`): **0 of 96**
  specification/inference/threshold combinations pass on a quantity outcome.
  The 2 passes are both the wage first stage against an "elasticity 1"
  threshold, which there means full pass-through. That is not a meaningful
  gate.

**Which error process is realistic?** In the pre-period, after county and
former-region × FY FE, the share of residual variance common to a state-FY
cell is 1.7% for positions and 0.8–1.5% for hours. Pure county noise would
give 1.6% (`$S/track1_state_year_share.csv`, `04b_track1_preperiod.R`). The
data therefore look like the county-only process, not the 50%-state
processes. Under it:

- State CRV1 still over-rejects (0.16), because a few large states carry the
  treatment.
- The size-valid jackknife gives an elasticity MDE of **1.5 (FY2026) and 1.2
  (with FY2027)**, close to the 1.0 gate but not through it.
- In the registry's units this is 0.013 and 0.0104 against a threshold of
  0.01.

This uses the observed state share to choose among the declared processes. It
is a reading of the power results, not a change to the declared gate, which
requires every shared process to pass.

### Pre-period diagnostics (pre-FY2026 data only)

These diagnostics were declared in the registry and run after it was frozen
(`04b_track1_preperiod.R`; `$S/track1_pre_trend_tests.csv`,
`$S/track1_pre_linear_trend.csv`, `$S/track1_fy2025_growth_check.csv`). The
dose is the FY2026 statewide rate minus the FY2025 regional rate.

- **The dose predicts a long, smooth pre-trend.** On positions (Q1–Q3), the
  dose × FY coefficients (reference FY2025) rise monotonically from −0.0152
  in FY2012 to −0.0049 in FY2019 and 0.0005 in FY2024. No single coefficient
  is significant.
  - The joint test of FY2012–FY2024 rejects for every outcome and dose
    (positions F = 4.6, p = 6e−5; hours F = 7.3, p = 2e−7). A CRV1 Wald test
    of 13 restrictions with 46 clusters is known to over-reject, so read this
    with care.
  - The one-parameter linear drift is 0.00125 (0.00115) per dollar of dose
    per year, p = 0.28 and wild p = 0.47.
- **What the drift does to the declared estimand.** States whose new
  statewide rate is high relative to their former region (high farm wages) had
  been gaining H-2A use. This is the threat the plan names ("the dose comes
  from state OEWS wages"). The registry's primary model compares FY2026 with
  the FY2012–FY2025 average. If the drift continues, that comparison carries
  a bias of about **+0.009 positions per 2011 farm worker per dollar**, almost
  the registry's 0.01 threshold and about 1.0 in elasticity units at the
  calibration means. That is as large as the realistic MDE. Hours show
  the same pattern (bias about 5–7.5 hours per dollar against a mean of 82).
- **No FY2025 warning sign.** The within-region slope of FY2024→FY2025
  growth on dose is −0.0005 (0.0014), p = 0.70. That ranks 11th of 13
  adjacent-year slopes by |t|. Hours give −1.73 (2.23), 7th of 13. Nothing
  suggests a 2025 surge correlated with dose, though enforcement-driven
  demand could still appear in FY2026 itself.

### Candidate amendments (B adopted as primary on 2026-11-02; see below)

Because the drift was found before any post outcome was read, the
specification can still be amended without looking at results.
`04c_track1_amendment_power.R` simulates two candidates with the same DGPs,
seed and gate (`$S/track1_amend_decision.csv`):

| Candidate | Error process | Positions: state-CRV1 MDE (elasticity) | Hours: state-CRV1 MDE (elasticity) |
| --- | --- | --- | --- |
| A. State linear trends, FY2012–FY2026 | county only | 0.016 (1.9) | 12.4 (1.4) |
| A. Same | shared state shocks | 0.086 (10.0) or beyond the grid | 65–83 (7.3–9.3) or beyond |
| A. State trends, through FY2027 | county only | 0.015 (1.7) | 11.3 (1.3) |
| B. FY2025 and FY2026 only | county only | 0.0080 (1.1) | 7.1 (0.9) |
| B. Same | shared state shocks | 0.042–0.060 (5.7–8.2) | 38–54 (4.9–7.0) |

These candidate runs computed state CRV1 and former-region CRV1 only. State
CRV1 again over-rejects (null 0.14–0.18), so its MDEs **understate** the
size-valid MDE. In the main run the jackknife MDE was about 1.5 times the
CRV1 MDE.

- **Neither amendment passes.** State trends, which remove the drift, lift the
  realistic MDE to about 2 (likely about 2.5–3 once size-corrected).
- **The FY2025-vs-FY2026 contrast** limits the drift to one year (about
  0.0013 per dollar, or 0.15 elasticity units, if it continues). Its
  county-only CRV1 MDE is about 1.1, so a size-valid version is plausibly
  about 1.5.

So the 2025 rule gives, at best, an elasticity MDE of about 1.5 after one post
year. The trend threat pushes it toward 2–3 unless you accept a short-window
contrast.

### Threats

- **Dose built from state OEWS wages:** confirmed as a pre-trend (above).
  State linear trends (amendment A) are the declared remedy, and the real
  state minimum wage is a declared control.
- **Concurrent 2025 shocks that differ by state**, above all immigration
  enforcement, which may raise H-2A demand. Nothing in the repository measures
  enforcement. This is the main unaddressed threat. The FY2025 growth check is
  clean, but an FY2026 shock correlated with dose cannot be ruled out.
- **Legal status:** confirmed by you on 2026-11-02 (rates in force throughout
  FY2026 Q1–Q3).
- **Skill level is an employer choice**, so the statutory skill-I rate is the
  intent-to-treat treatment.
- **Transition:** job orders filed before 2025-10-02 carry the old rate and
  attenuate the intent-to-treat slope. The cohort-aligned sensitivity is
  declared, and the simulation assumes full treatment.
- **The H-2A housing adjustment and the new skill structure** change
  compensation in other ways at the same moment. The H-2A skill-I sensitivity
  partly addresses this.

### Decisions and amendment (2026-11-02)

After reading the stop-point results you decided:

- **Read FY2026 and estimate.**
- **Amend before reading.** The FY2025-vs-FY2026 contrast (candidate B) is
  primary. The frozen registry and state trends (candidate A) are declared
  sensitivities.
- **Legal status.** You confirmed that the statewide rates were adopted after
  full notice and comment, that legal challenges ultimately failed, and that
  the rates applied throughout FY2026 Q1–Q3. The base registry called the
  rule an interim final rule. That description was written from memory, and
  your statement governs.

The amendment is `explorations/identification/track1_registry_amendment1.json`,
frozen before any FY2026 row was read. **SHA-256
`433264ec85845f2ddb97c291073c19585ca9b80e2bb137aa0915fe38e29799e9`.** It
references the base registry's hash, which is unchanged. It also makes the
state jackknife the primary inference, because state CRV1 failed the size gate
in every simulated error process. That choice was made before any outcome was
read and is labeled.

### Post-policy estimates

Script `08_track1_estimate.R`; outputs `$S/track1_post_*.csv`. It hash-checks
both registries before reading FY2026.

- **Outcome rebuild checks out.** The FY2012–FY2025 rebuild matches the
  frozen pre-period panel to within 2e−12 (the script stops otherwise). In
  the event study, the FY2026 coefficient equals the primary contrast to every
  printed digit.
- **FY2026 data.** Decisions run to 30 June 2026, so the full Q1–Q3 window is
  present.
  - Q1–Q3 certified positions in the sample counties rose from 250,732
    (FY2025) to 280,538. The FE absorb that national change.
  - Only 6,002 FY2026 Q1–Q3 positions (2.1%) come from cases received before
    2025-10-02, so transition attenuation is small.
  - Only 8 decisions are dated October 2025, against 1,089 in October 2024,
    with more than usual in November–December. That looks like the October–November
    2025 federal funding lapse. Delayed decisions still fall inside the
    October–June window.

**Primary (B), U.S. skill-I rate, 46 states, 2,921 counties.** "Wild" is the
restricted Webb bootstrap over states; "RI" permutes FY2026 statewide rates
across states within a former region (9,999 draws each).

| Outcome | Estimate per statutory $ (jackknife SE) | Jackknife p | State CRV1 p | Wild p | RI p | Elasticity [95% jackknife] | Elasticity MDE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Positions per 2011 farm worker | 0.0038 (0.0067) | 0.58 | 0.32 | 0.51 | 0.45 | 0.40 [−1.06, 1.86] | 2.03 |
| Hours per 2011 farm worker | 2.05 (5.17) | 0.69 | 0.47 | 0.59 | 0.52 | 0.21 [−0.85, 1.27] | 1.48 |
| PPML positions (count) | −0.0076 (0.021) | 0.72 | 0.51 | — | — | −0.12 [−0.77, 0.53] | 0.90 |
| PPML hours (count) | −0.0129 (0.026) | 0.62 | 0.34 | — | — | −0.20 [−1.00, 0.61] | 1.12 |
| First stage: offered hourly wage | 0.66 (0.095) | 1e−8 | 4e−14 | 0.0001 | 0.0001 | pass-through 0.66 [0.47, 0.85] | — |

- **No detectable quantity response, and the first stage is strong.** The
  rule lowered wages; the coefficients are slopes, and a positive slope means
  a larger cut gives a larger fall.
  - **Statutory rate:** the U.S. skill-I rate fell in every sample county,
    by $4.09 on average ($17.4 → $13.3).
  - **Offered wages:** among the 2,686 counties with hourly offers in both
    years, the hours-weighted offered wage fell in 90% of them. The mean fell
    from $17.7 to $15.5 (−$2.18).
  - **Larger cuts brought larger falls:**
    - deepest-cut third of counties: rate −$5.32, offered wage −$2.81;
    - middle third: −$4.10, −$2.24;
    - shallowest-cut third: −$2.85, −$1.48.
  - **Slope:** within former regions, offered wages fall 66 cents per
    statutory dollar of cut.
    - Added 2026-11-13: 62 of the 66 cents are the same employer in the
      same county offering less; entry, exit and reweighting add 5 cents
      (`markdowns/track1_employer_decomposition.md`).
  - **Per dollar of actual wage:** a dollar of offered wage corresponds to
    about 1.5 statutory dollars, so the reduced-form effects and intervals
    scale up by about 1.5 in levels.
  - **Quantity coefficients:** these are slopes too. The positive positions
    point estimate means counties with larger cuts saw slightly smaller
    relative H-2A growth, which is not significant.
- **The equal-county ratio outcomes miss the 1.0 gate.** The realized MDEs are
  2.0 for positions and 1.5 for hours. The simulation's realistic value was
  about 1.5, so realized noise for positions was somewhat larger.
- **Size-weighted PPML is the most precise row.** Its realized MDE is 0.90
  for positions, which would pass the 1.0 gate. But PPML was declared
  secondary, and the gate was defined on the primary outcome. Report it as
  secondary.
- **Placebo (FY2024 vs FY2025, FY2026 dose): clean.** Positions −0.0005
  (0.0023), elasticity −0.06 [−0.60, 0.48]. Hours −1.7 (3.7). PPML −0.11 and
  −0.08. First stage 0.003 (0.057). Every jackknife, wild and RI p-value is
  above 0.5. The one exception is former-region CRV1 for hours (p = 0.04),
  with 15 clusters.

**Declared sensitivities** (positions; hours are in the CSV and show the same
pattern):

| Specification | Elasticity [95% jackknife] | Elasticity MDE | Note |
| --- | --- | --- | --- |
| S3 H-2A skill-I rate | 0.10 [−0.88, 1.09] | 1.37 | first stage 0.65 (0.13) |
| S3 U.S. skill-II rate | 0.81 [−1.61, 3.23] | 3.37 | first stage 0.82 (0.12) |
| S4 cohort-aligned rate | 0.37 [−0.79, 1.54] | 1.62 | no RI (county-varying treatment) |
| S5 + general state minimum wage | 0.34 [−1.25, 1.93] | 2.20 | 39 states (see note) |
| S5 + applicable ag minimum (current coding) | 0.38 [−1.08, 1.83] | 2.02 | 39 states |
| S1 frozen static, FY2012–FY2026 | 1.77 [−3.67, 7.21] | 7.57 | carries the drift (§3 pre-period) |
| S1 frozen static, PPML positions | −0.25 [−2.21, 1.70] | 2.72 | |
| S2 state linear trends | 0.32 [−0.90, 1.54] | 1.70 | no wild bootstrap (fixest slopes) |

- **The frozen static model's point estimate (1.77) is where the drift
  points.** With trends (S2) or the one-year contrast (B), it falls to about
  0.3–0.4.
- **S5 lost seven states** (AL, GA, LA, MS, SC, TN, WY). The FRED minimum-wage
  artifact has no FY2025/FY2026 rows for states whose floor is the federal
  minimum. This is a producer gap in `code/utilities/pull_fred_minimum_wages.R`,
  not a coverage decision.
- **Event study:** the FY2012–FY2024 coefficients are the pre-period values in
  §3 above. FY2026 is 0.0038 (0.0037), state CRV1.

**Reading.** The 2025 statewide rule gives a clean, strong first stage and a
precise placebo. It shows no detectable H-2A quantity response to
within-region differences in the new statutory rate.

- **What it rules out:** positions elasticities below about −1.1 (equal-county
  ratio) or −0.8 (size-weighted PPML), and hours elasticities below about
  −0.9 to −1.0. These are per statutory dollar, one season after the change.
- **What it does not rule out:** conventional own-wage labor-demand
  elasticities, about −0.15 to −0.75 in Hamermesh's (1993) survey. That range
  is cited from memory and was not checked here.
- **The main caveats stand:**
  - An unmeasured FY2026 shock correlated with the dose, especially
    immigration enforcement.
  - One season only. Contracts and crop choices adjust over longer horizons.
  - The identifying variation is a cut for most states (mean dose −$3.96), so
    the estimate is a response to a fall in the rate.

## 4. Track 2: domestic labor cost via state minimum wages

Script `05_track2_minwage.R`; outputs `$S/track2_*.csv`. Outcomes are the
retained certified positions and hours per 2011 farm worker, 2011–2025, equal
county weights, clustered by state (48 clusters).

### Agricultural-exemption audit (done first)

`$S/track2_exemption_audit.csv`, `$S/track2_exemption_audit_meta.csv`:

- **The producer has no source notes or dates.** `a01/07_state_minimum_wages.py`
  hard-codes 18 exempt states with no citation and no effective dates: AK, DE,
  GA, IN, KS, KY, MA, ME, NE, NH, NJ, OK, RI, VA, VT, WA, WV and WY.
  `c01/04_state_minimum_wages.R` takes the 2024 value for every year and
  treats states missing from the 2024 rows as exempt. That adds AL, LA, MS, SC
  and TN, which has no effect because none of them is above the federal
  minimum.
- **No state's coverage changes over time in the producer**, so its source
  notes cannot show which states' coverage changed during 2008–2025.
- **The classification moves the applicable floor in 11 states:** WA, MA, ME,
  RI and VT in all 18 years; DE and NJ in 14; WV in 13; NE in 11; VA in 5; NH
  in 2. The largest suppressed gap is $9.41 (WA).
- **Some classifications look wrong.** From domain knowledge, *not verified
  against statute and not checked online*, WA and NJ minimum wages are
  believed to cover farmworkers. MA has a separate $8 agricultural minimum. VA
  coverage may have changed in 2020, and WV may be a small-employer exemption.
- **About 14% of the agricultural floor's identifying (residual) variance**
  comes from these contested states (21% for the general minimum).
- **Results barely move** when the contested states are dropped or the
  general state minimum is used instead (below).
- **Producer fix (implemented 2026-11-02 after your approval).** See "Coverage
  fix" below.

### Coverage fix: statute research and corrected estimates

**Research.** On 2026-11-02 I compiled, by web research, a dated coverage
schedule for all 50 states and DC over 2008–2026, with a citation, URL and
confidence level for each spell. It is now the hand-maintained source table
`code/a01_sources/state_ag_minwage_coverage.csv`, with notes in
`state_ag_minwage_coverage_notes.md`. Coding: an exemption that copies a
federal one (500 man-day small farms, local hand-harvest commuters, family,
range) still counts as covered, because it does not reach H-2A-size housed
crews.

**Findings against the old list:**

- **WA:** covered throughout. Its exemption was from overtime.
- **RI:** covered throughout. Its agricultural exemption is overtime-only.
- **NJ:** covered at the general minimum until 2019-06-30. After that, a
  separate, slower farm schedule applies under P.L.2019 c.32, starting at
  $8.85 and rising to $14.20 in 2026.
- **MA:** an $8.00 farm minimum from 2015. Before that the statutory $1.60
  never binds.
- **ME:** exempt through 2025. Covered from 2026 under P.L. 2025 c.232.
- **Confirmed exempt:** AK, DE, GA, IN, KS, KY, NE, NH, OK, VA, VT, WV and WY,
  plus the states with no minimum wage law.
  - VA's 2020 amendments kept the farm exclusion. A 2026 law removes it from
    2027, but H-2A workers stay excluded.
  - WV's is a genuine agricultural exclusion.
- **Missed by the old list:** WI was exempt until 2009-07-23. NC is exempt.
  PA, HI, NM and MI (2014–16) are partial.
- **Shakiest row:** CO 2008–2021 (medium confidence). TX is low but moot. AK,
  IL and HI rely on secondary hosts.

**Producer changes:**

- `code/c01_clean/04_state_minimum_wages.R` now reads the dated table. The
  coverage spell in effect on 1 July sets the floor:
  - state minimum → max(state, federal);
  - federal minimum → federal;
  - farm-specific rate → max(farm rate, federal).
- Partial coverage without a separate rate (HI, MI 2014–16, NM, PA) is coded
  at the state minimum, because the exclusions do not reach hourly H-2A-size
  crews.
- It adds `ag_floor_rule`, `ag_specific_min_wage` and `ag_coverage_backcast`.
  The last flags years before 2008, which take the first spell.
- It drops the "2024 value for all years" join and the default to exempt.
- `code/a01_sources/07_state_minimum_wages.py` no longer hard-codes the list.
- `.gitignore` now excepts the coverage table (and the overtime schedule) from
  its `*.csv` rule, as it already does for `aewr_policy_events.csv`.
- Docs updated: `code/a01_sources/README.md` and
  `content/architecture/shared-panel-generation.md`. Agent context was
  regenerated; `agent_docs.py verify` and its tests pass. `zola` could not run
  in this environment (library symbol error).

**Validation and what was not regenerated:**

- I ran the changed c01 script with its single write redirected to scratch
  (`$S/state_real_minwages.fixed.parquet`). Every non-agricultural column is
  identical to the current artifact.
- The agricultural floor changes only in ME (2026), MA (2015+), NJ, NC, RI,
  WA and WI (2008–09).
- **Regenerated on your instruction (2026-11-02):** `c01/04`, then the
  `c01/13` merge, the `c02/01` panel build and the `c02/02` validation, which
  passes.
  - The regenerated artifact is identical to the validated scratch build.
  - In the merged and shared panels only the farm-floor columns changed:
    `agriculture_exemption`, `prevailing_ag_min_wage`, its PPI version and
    their `_l1` lags. The changes fall in MA, NJ, NC, RI, WA and WI over
    2008–2025; ME's change is in 2026, outside the panel.
  - Three columns and their lags were added: `ag_floor_rule`,
    `ag_specific_min_wage` and `ag_coverage_backcast`.
  - Row keys are unchanged, and every other column is identical.
  - Backups of the three prior artifacts are in the session scratchpad
    (`mw_rebuild_backup/`).
- **Still stale:** the Mundlak–Chamberlain branch's retained outputs, which
  use `prevailing_ag_min_wage` in its bite and fraction-affected treatment
  coordinates. It has not been rerun.
- The a01 output keeps its stale `agriculture_exemption` column until that
  credentialed FRED step is rerun. Nothing reads it now.

**Track 2 with corrected coverage** (`05_track2_minwage.R <out> <fixed parquet>`,
outputs `$S/track2c_*.csv`). Spec (1), state CRV1:

| Row | Old coding: elasticity [95%], MDE | Corrected coding: elasticity [95%], MDE |
| --- | --- | --- |
| Positions on applicable ag floor | −0.10 [−0.57, 0.37], 0.65 | 0.09 [−0.38, 0.56], 0.65 |
| Hours on applicable ag floor | 0.03 [−0.40, 0.46], 0.60 | 0.26 [−0.16, 0.68], 0.59 |
| Lead (positions) | −0.13 [−0.56, 0.31] | 0.17 [−0.30, 0.63] |
| Translated to AEWR, positions | 0.16 [−0.58, 0.91], 1.04 | −0.14 [−0.88, 0.60], 1.03 |
| Translated to AEWR, hours | −0.05 [−0.74, 0.64], 0.96 | −0.41 [−1.07, 0.26], 0.92 |

- **Still a well-powered null for the domestic floor.** The corrected event
  study is clean (joint pre-period p = 0.78 for the ag floor).
- **The mechanical AEWR translation moves** from about −0.75 to −0.9
  (positions) and −1.1 (hours) at its lower end. The next subsection
  explains why this is not an independent bound.
- **The old code path is unchanged.** Run without the second argument, the
  script reproduces every original `track2_*.csv` byte for byte.
- **The "excluding contested states" row is no longer meaningful** under the
  corrected coding, because "contested" was defined by the old list.

### Estimates

β per real 2012 dollar. The bootstrap is a restricted Webb wild bootstrap
over states, 9,999 draws. Mean real agricultural floor $8.20; mean outcome
0.094 positions and 91.5 hours.

| Specification | Outcome | β (SE) | Wild p | Elasticity [95% CI] | Elasticity MDE | MDE % mean |
| --- | --- | --- | --- | --- | --- | --- |
| (1) Agricultural floor, county + region × year FE | positions | −0.00118 (0.00266) | 0.73 | −0.10 [−0.57, 0.37] | 0.65 (0.82 region CRV1) | 8.0 |
| (1) Same | hours | 0.342 (2.405) | 0.90 | 0.03 [−0.40, 0.46] | 0.60 | 7.4 |
| (1) General state minimum | positions | −0.00170 (0.00288) | 0.67 | −0.15 [−0.68, 0.37] | 0.73 | |
| (1) General state minimum | hours | −1.12 (3.19) | 0.81 | −0.10 [−0.69, 0.49] | 0.82 | |
| (1) Contested states dropped (37 states) | positions | −0.00139 (0.00307) | 0.73 | −0.12 [−0.64, 0.41] | 0.72 | |
| (2) Floor × frozen p10 closeness | positions | −0.00111 (0.00080) | 0.22 | differential | 0.20 | |
| (2) Same, plus exposure × year FE | positions | −0.00569 (0.00412) | 0.18 | differential | 1.01 | |
| (2) Same, plus exposure × year FE | hours | −5.72 (3.36) | 0.096 | differential | 0.84 | |
| (3) Within-region cross-state pairs, unscreened (38 pairs), pair × year FE | positions | 0.00201 (0.00441) | 0.79 | 0.22 [−0.74, 1.17] | 1.32 (1.76 pair CRV1) | 16 |
| (3) Same | hours | 3.81 (4.25) | 0.55 | 0.38 [−0.47, 1.22] | 1.18 | |
| (3) Screened pairs (22) | positions | −0.00423 (0.00571) | 0.64 | −0.41 [−1.53, 0.71] | 1.54 | |

Pre-trends and placebos:

- **Lead placebo.** Next year's floor gives −0.00137 (0.00232), wild p 0.64,
  on positions and −2.13 (1.88), wild p 0.39, on hours. The current-year
  terms stay near zero.
- **Event study around large increases** (nominal increase of at least 5%,
  declared before fitting). Joint pre-period p-values are 0.75 (positions) and
  0.84 (hours) for the agricultural floor, and 0.65 and 0.78 for the general
  minimum. Post-event coefficients are small, negative and insignificant
  (`$S/track2_event_study.csv`).
- **Bite sign.** The bite interaction (2) is negative. That is the wrong sign
  for substitution toward H-2A where the floor binds on domestic workers,
  though it is not significant.

### Relative-price restriction (spec 4)

Without region × year FE (county + year FE), y is regressed on the real AEWR
and the real agricultural floor jointly:

- **Positions:** AEWR −0.0367 (0.0255), floor +0.0049 (0.0067). The test of
  equal and opposite coefficients gives p = 0.22 (state clusters) and 0.23
  (region).
- **Hours:** AEWR −29.9, floor +4.08, Wald p = 0.14.

The test has no power, because the AEWR term is as imprecise as in the plain
panel. Failing to reject is not support for the restriction.

**Translation.** Suppose only AEWR minus the domestic floor matters, and a $1
rise in the floor raises the cost of the competing domestic labor by $1. Then
the AEWR slope is −β_floor. At the mean real AEWR of $13.06, spec (1) implies
an AEWR elasticity of **0.16 [−0.58, 0.91]** for positions and **−0.05
[−0.74, 0.64]** for hours. The implied MDE elasticities are 1.04 and 0.96
(`$S/track2_relprice_translation.csv`).

**Why the translation is not an AEWR bound (corrected 2026-11-02).** The
translation is a chain of three links. Only the first is estimated.

1. **Minimum wage → H-2A use.** Estimated and credible (above).
2. **Minimum wage → competing domestic wage (pass-through λ).** Not
   estimated; λ = 1 was assumed.
   - The floor averages $8.20 against an AEWR of $13.06. A $1 rise in the
     floor raises domestic cost by $1 only for workers paid near the floor,
     so λ < 1 is likely.
   - The implied AEWR elasticity is the translated value divided by λ, and
     its interval widens by 1/λ.
3. **Cross-price effect → own-price effect ("symmetry").** Equal and
   opposite effects hold only at fixed output.
   - With two inputs and a scale (product-demand) elasticity η ≥ 0, the
     own-price elasticity of H-2A use equals minus the cross-price elasticity
     with respect to the domestic wage, minus η.
   - A higher domestic wage shrinks output and so lowers H-2A use, which
     offsets substitution toward H-2A. The sign-flipped minimum-wage effect
     therefore understates how negative the AEWR effect is, by η.

**It is the bite design's model seen from the other side.** Both assume H-2A
use depends on the AEWR-minus-domestic-wage gap. The bite design varies the
AEWR across counties with different frozen gaps; Track 2 varies the domestic
floor with the AEWR held fixed. Both rely on an unobserved domestic
alternative wage. The bite report found that no measure of it predicts where
the AEWR binds: 94% of 2011–13 hourly H-2A hours were offered at exactly
the AEWR, and within-region R² is below 0.002 for every measure.

**What would rescue it:**
- **Estimate λ.** Regress local domestic farm wages (QCEW crop and livestock
  pay per job, OEWS farm occupations, ACS state farm wage) on the farm floor
  with the same fixed effects.
- **Bound η.** Test whether farm output or acreage responds to the floor.
- **Test the gap restriction** (spec 4). This needs a precise AEWR effect on
  the other side.

### Threats

- **Endogenous timing.** Minimum-wage increases follow state economic and
  political conditions. The flat leads and event-study pre-periods are the
  main defense, and the pairs are too imprecise to add much.
- **Scale effects.** A higher floor also changes farm scale and output, so the
  response is not pure substitution.
- **Misclassified exemptions.** They are not dated, as audited above.

## 5. Track 3: multi-region employers

Script `06_track3_multiregion.R`; outputs `$S/track3_*.csv`.

- **Sample (declared before estimation):** employer-years with positive
  positions in at least two AEWR regions. Cells are every region the employer
  used in any multi-region year, crossed with its multi-region years, with
  zeros kept.
- **Model:** outcome is the region's share of the employer-year's positions
  or hours, or log1p(positions). Treatment is the real regional AEWR. FE are
  employer × year and employer × region. 0.46% of positions could not be
  placed in a region.
- **Identification is still region-year.** Employer × year FE remove employer
  demand and composition, but regional crop shocks remain. Region price and
  quantity indices barely change the estimates (2011–19 position share
  −0.0265 (0.0193) with controls).

| Calendar, window | Outcome | β (region SE / two-way SE) | Wild p | Elasticity [95% CI] | Elasticity MDE | MDE % mean |
| --- | --- | --- | --- | --- | --- | --- |
| Start year, 2011–19 | position share | −0.0298 (0.0229 / 0.0203) | 0.24 | −1.19 [−3.14, 0.75] | 2.57 | 19.6 |
| Start year, 2011–19 | hours share | −0.0206 (0.0233 / 0.0210) | 0.46 | −0.83 [−2.81, 1.16] | 2.62 | 20.0 |
| Start year, 2020–24 | position share | +0.0188 (0.0127 / 0.0130) | 0.085 | 0.67 [−0.29, 1.63] | 1.26 | 9.1 |
| Start year, 2020–24 | hours share | +0.0115 (0.0114 / 0.0125) | 0.26 | 0.41 [−0.45, 1.26] | 1.13 | 8.2 |
| Start year, 2011–24 | position share | −0.0219 (0.0174 / 0.0160) | 0.32 | −0.84 [−2.24, 0.57] | 1.86 | 13.7 |
| Fiscal year, 2011–19 (calendar sensitivity) | position share | −0.0408 (0.0183 / 0.0167) | 0.022 | −1.65 [−3.22, −0.08] | 2.07 | 15.8 |
| Fiscal year, 2011–19 | hours share | −0.0335 (0.0178) | 0.038 | −1.35 [−2.88, 0.17] | 2.02 | 15.3 |

Start-year windows stop at 2024. The FY ≤ 2025 read leaves start-year 2025
incomplete.

- **Lead placebo.**
  - 2011–19: the next-year AEWR term is −0.0153 (0.041), wild p 0.72.
  - 2020–24: the lead is +0.0157 (0.0095), wild p 0.13, on position shares,
    and +0.0208 (0.0097), wild p 0.089, on hours shares. On hours the lead is
    larger than the current-year term (0.0074), the same lead problem as in
    the panel designs.
- **The sign flips between eras** (negative for 2011–19, positive for
  2020–24). Using positive cells only flips the 2011–19 sign to +0.013
  (0.018).
- **Only the fiscal-year calendar sensitivity** reaches wild p < 0.05. The
  primary start-year version does not (0.24).

## 6. Track 4: state farmworker overtime laws

You approved acquiring the schedule on 2026-11-02. The specification below
was written before acquisition. Results follow it.

- **Dose.** For job order j in state s with weekly hours h_j (`number_of_hours`):
  d_j = 0.5 × max(0, h_j − θ_{s,t,size}) / h_j. Here θ is the weekly overtime
  threshold in effect at the job's start date for the employer's size class.
  Use daily thresholds where a state has them.
- **Frozen exposure.** County (or employer) exposure is
  D_ct = Σ_j w_j^base × 0.5 × max(0, h_j − θ_st)/h_j. The weights w^base are
  the certified-hours mix of weekly hours in a pre-reform window, FY2015–2018,
  before the earliest phase-in. Time variation then comes only from θ_st, not
  from employers' current choice of weekly hours.
- **Specifications:**
  - (i) Certified hours or positions per 2011 farm worker on D_ct, with county
    and former-region × year FE. This uses threshold changes across states
    within a region.
  - (ii) The same with state × year FE. This uses only differences in
    baseline hours mix within a state-year. It is a heterogeneity design,
    like bite 1(b).
  - (iii) PPML counts. (iv) An employer-level version with employer and
    region × year FE.
- **Estimand.** The H-2A hours response to an increase in hourly cost
  concentrated on long-hours contracts. Domestic workers at the same farms
  also become more expensive, so it is not the AEWR effect. It is close to
  the hours response to hourly cost that the welfare model needs.
- **Data needed:** state; effective dates; weekly and daily thresholds;
  employer-size classes and their phase-in schedules; whether H-2A contracts
  disclose overtime pay; statutory sources.
  - Candidates to verify at acquisition: CA (multi-year phase-in from 2019),
    WA, NY, OR and CO (2020s phase-ins), and states with long-standing
    thresholds such as MN and MD. This list is from memory and was not checked.
- **Support check to run first:**
  - the number of states and state-years with a threshold change;
  - the share of baseline certified hours on job orders above each new
    threshold;
  - the within-state-year SD of D_ct;
  - the number of counties with D_ct > 0.
- **Inference.** With about five treated states, CRV1 is unreliable. Use
  randomization inference over states (Conley–Taber; Ferman–Pinto). Report
  the wild cluster restricted bootstrap with MacKinnon–Webb's few-treated
  caveats, and a synthetic-control or synthetic-DiD comparison for each
  treated state.

### Acquired schedule

The schedule is `explorations/identification/sources/state_ag_overtime_schedule.csv`,
with method and source notes in `state_ag_overtime_notes.md` alongside. It has
84 rows covering every state and DC, 2008–2027, each with a citation, URL and
confidence level. It was compiled by web research on 2026-11-02.

- **Farm-worker overtime thresholds exist in 12 jurisdictions.**
  - **Phase-ins within the window:**
    - **CA:** before 2019, 10 hours a day plus a 7th-day rule. Then 55/50/45/40
      weekly and 9.5/9/8.5/8 daily, from 2019 for employers with 26 or more
      employees and three years later for smaller ones.
    - **NY:** 60 hours (2020), 56 (2024), 52 (2026), falling to 40 by 2032.
    - **WA:** 55/48/40 over 2022–24, and dairy at 40 from 2020-11-05.
    - **OR:** 55 (2023), 48 (2025), 40 (2027).
    - **CO:** 60 hours from 2022-11-01. Employer-class thresholds of 48, 54
      or 56 in 2024, and 48 in 2025–26. SB26-121 raises it to a uniform 56
      from 2027.
    - **VA:** 40 hours, 2021-07-01 to 2022-06-30 only.
  - **Unchanged throughout:** MN (48), MD (60, large employers), HI (40, 20 or
    more employees) and DC.
  - **Narrow carve-outs only:** MA covers post-harvest packing and ME covers
    large egg operations.
- **Corrections to my earlier list from memory:** Colorado's schedule and
  Washington's dairy date were wrong.
- **Low or medium confidence:** HI, VA 2021–22, the MA and ME carve-outs, KY's
  7th-day rule, and several no-overtime rows that rest on secondary sources.
  The CA, CO and HI statute pages were read through mirrors or fiscal notes.
- **H-2A job orders must disclose state-required overtime.** This is ETA-790A
  item A.8g plus Addendum C, and 20 CFR 655.122(l)(4) from the 2024 rule.
  Enforcement of that rule was suspended on 2025-06-20, and its proposed
  rescission keeps the disclosure. A separate injunction of the 2024 rule is
  noted from memory only.

### Support and estimates

Script `09_track4_overtime.R`; outputs `$S/track4_*.csv`. Coding rules were
fixed in the script header before any outcome regression:

- the threshold in effect on 1 July of the start year;
- crop and general agriculture rows only;
- CA's 26-or-more class, CO's "other employer" class in 2024 (sensitivity:
  "highly seasonal"), MD and HI large-employer classes;
- daily thresholds converted with a 6-day week;
- counties with no FY2015–18 job orders use their state's baseline hours
  mix.

Outcomes are the retained start-year ratios for 2011–2025, plus PPML counts.

**Support:**
- Six treated states (CA, CO, NY, OR, VA, WA) with 15 state-year changes.
  344 counties are in treated states and 229 ever have positive exposure.
- Only 20% of baseline certified hours are on job orders listing more than 40
  weekly hours, 4% more than 48 and 0.1% more than 60. NY's 60/56 thresholds
  therefore bind almost nowhere.
- Exposure is small: mean D in treated states in 2025 is 0.012, roughly a
  1.2% rise in hourly cost. CA reaches 0.043, with a within-state-year SD of
  0.031 in treated post periods.

| Specification | Positions: estimate (state CRV1 SE) | Wild p | RI p | Elasticity w.r.t. hourly cost [95%] |
| --- | --- | --- | --- | --- |
| (i) county + former-region × year FE | −0.76 (0.11) | 0.068 | 0.13 | −7.6 [−9.9, −5.3] |
| (ii) county + state × year FE | −0.72 (0.08) | 0.036 | 0.18 | −7.2 [−8.8, −5.6] |
| (i) with CO "highly seasonal" class | −0.74 (0.10) | 0.053 | — | −7.4 [−9.6, −5.3] |
| Lead check: D | −0.21 (0.26) | 0.80 | — | −2.1 |
| Lead check: **D one year ahead** | **−0.61 (0.20)** | **0.035** | — | **−6.1** |
| (iii) PPML counts | **+8.9 (2.2)** | — | — | **+8.9 [4.4, 13.5]** |

Hours give the same pattern: (i) −8.2, a lead of −5.8 with wild p 0.065, and
PPML +3.1.

**Verdict: not credible as an average-effect design.**

- **Exposure predicts outcomes one year early.** The lead coefficient carries
  most of the effect, and the contemporaneous term falls to −0.21 once the
  lead is included. So the negative ratio estimate reflects pre-existing
  trends in the treated states' small counties, not a response to overtime.
- **Size-weighted PPML has the opposite sign.** Large treated counties,
  mostly in CA and WA, grew fast through the phase-ins.
- **With six treated states, randomization inference does not reject.** RI
  p-values are 0.13–0.18, and the cluster-robust p-values of 1e−8 are not
  trustworthy.
- **Mechanically, the elasticities are large because the dose is tiny.** A
  1% cost change moves little, so any trend in treated states maps into
  elasticities near ±8.

Keep the schedule as a sourced input; it is useful for the welfare model's
hours margin. Do not use Track 4 as an identification design.


## 7. Synthesis: what each design rules out

`07_synthesis.R`, `$S/synthesis.csv`. Elasticities use each design's own
sample means. 95% intervals are CRV1 t-intervals, Anderson–Rubin for the panel
IV, and the region pairs bootstrap for PPML-IV. Wild bootstrap results are
p-values; the bootstrap was not inverted into intervals. **Estimands differ
across rows and are not pooled.**

| Design | Outcome | Estimand | Key assumption | Elasticity [95% interval] | Excludes |
| --- | --- | --- | --- | --- | --- |
| Plain panel OLS | positions / hours | average effect | region-year AEWR deviations exogenous | −6.3 [−18.0, 5.4] / −4.8 [−12.8, 3.2] | nothing below ±12 |
| Panel IV col. 4 | positions / hours | average effect (IV) | dissimilar-donor exclusion | AR [−31.8, 9.0] / AR [−22.8, 4.6] | nothing useful |
| PPML-IV (control function) | positions / hours | size-weighted semi-elasticity (IV) | same exclusion | [−6.7, 4.9] / [−5.0, 5.7] | nothing below ±5 |
| PPML, no IV | positions / hours | size-weighted semi-elasticity | region-year deviations exogenous | 1.00 [−0.59, 2.59] / 1.42 [−0.09, 2.92] | ε ≤ −0.6 / ε ≤ −0.1, **if** exogenous |
| Region-year aggregate OLS / IV | positions | elasticity of total use | exogenous / exclusion | 1.91 [−0.41, 4.22] / 4.42 [−1.38, 10.2] | OLS: ε ≤ −0.4 if exogenous; IV: little |
| Employer panel 2011–19 | log1p positions | baseline employers' semi-elasticity | exogenous | 0.01 [−1.12, 1.14] | ε ≤ −1.1 |
| Bite 1(b) | positions / hours | differential per 1 SD gap | parallel trends across frozen gap | 0.22 [−0.14, 0.57] / 0.20 [−0.15, 0.56] | differential effects > ±0.6 per SD |
| Bite 1(b) → average | positions / hours | average | β_c ∝ frozen gap (strong; gap model) | 0.26 [−0.17, 0.68] / 0.25 [−0.19, 0.68] | ε ≤ −0.2, ε ≥ 0.7, **only** under the gap model; the gap measure does not predict binding |
| Border v6 ring, baseline | positions / hours | average border effect | border continuity, 25 edges | −1.8 [−4.9, 1.3] / −2.6 [−6.2, 0.9] | nothing useful |
| Track 2 (1), domestic floor, corrected coverage | positions / hours | elasticity w.r.t. domestic floor | MW timing exogenous given region × year FE | 0.09 [−0.38, 0.56] / 0.26 [−0.16, 0.68] | floor elasticities beyond about ±0.6 |
| Track 2 translated to AEWR, corrected coverage | positions / hours | relative-price elasticity | gap model; λ = 1; no scale effect (η = 0) | −0.14 [−0.88, 0.60] / −0.41 [−1.07, 0.26] | nothing on its own: widens by 1/λ and shifts down by η; shares the bite's gap model |
| (old coding, superseded) Track 2 translated | positions / hours | as above | as above | 0.16 [−0.58, 0.91] / −0.05 [−0.74, 0.64] | — |
| Track 3, start year 2011–19 / 2020–24 | position share | within-employer regional reallocation | regional demand ⟂ AEWR deviations | −1.19 [−3.14, 0.75] / 0.67 [−0.29, 1.63] | little; signs disagree |
| Track 1 statewide rule, FY2025 vs FY2026 (primary) | positions / hours | ITT of statutory rate, within former region, one season | no dose-correlated FY2026 shocks | 0.40 [−1.06, 1.86] / 0.21 [−0.85, 1.27] (jackknife) | ε ≤ −1.1 / ε ≤ −0.9 |
| Track 1, PPML counts (secondary) | positions / hours | size-weighted ITT semi-elasticity | same | −0.12 [−0.77, 0.53] / −0.20 [−1.00, 0.61] | ε ≤ −0.8 / ε ≤ −1.0 |
| Track 1 placebo, FY2024 vs FY2025 | positions / hours | placebo | — | −0.06 [−0.60, 0.48] / −0.19 [−1.00, 0.62] | (passes) |
| Track 4 overtime (i) / PPML | positions | elasticity w.r.t. hourly cost | threshold timing exogenous; 6 treated states | −7.6 [−9.9, −5.3] / +8.9 [4.4, 13.5] | not credible: lead fails, signs disagree, RI p 0.13–0.18 |

What the combination can and cannot exclude:

- **No design that relies on a credible exclusion restriction excludes large
  responses.** The IV-based rows (panel IV, PPML-IV, aggregate IV) are
  consistent with elasticities anywhere from about −5 to +5 or wider.
- **The non-IV rows that exclude ε ≤ −1 are not evidence of a small
  response.** These are PPML, the aggregate and the employer panel. Their
  point estimates are positive, which is what upward bias from regional
  demand predicts, and that bias would make them exclude negative elasticities
  spuriously.
- **Only one design gives an independent, credible bound** (corrected
  2026-11-02).
  - **Track 1** (the 2025 statewide rule, FY2025 vs FY2026) needs no
    exclusion restriction beyond "no dose-correlated FY2026 shock". Its
    placebo is clean, and it excludes elasticities below about −1.1
    (equal-county positions) or −0.8 (size-weighted PPML).
  - **The bite conversion and the Track 2 translation are consistent with
    it, but they are not independent of each other.** Both assume only the
    AEWR-minus-domestic-wage gap matters, and both rest on an unobserved
    domestic alternative wage that predicts nothing about binding. The Track 2
    translation also needs λ = 1, and η = 0 (no scale effect), which biases
    it toward zero.
  - **Nothing here excludes conventional labor-demand elasticities** of
    −0.15 to −0.75, and Track 1 cannot exclude a positive response.
- **Do not pool.** Estimands differ: average county effect, size-weighted
  effect, differential effect, relative-price elasticity, and an ITT on a
  2025 cut. The rows sit side by side.

## 8. Recommendations, integration and decisions

### Recommendation by track

| Track | Recommendation | Why |
| --- | --- | --- |
| 1. 2025 statewide rule | **Lead design for a lower bound.** Report the amended primary, placebo and all declared sensitivities. FY2026 Q4 and FY2027 data do not exist yet, so its extension on existing data (hours, requested positions, occupation and housing-adjustment dimensions, revealed binding) is strategy A1 in `markdowns/identification_search_agent_plan.md` | Strong first stage, clean placebo, 46 state clusters, assumption not about region-year deviations. One season; MDE 2.0 (primary) / 0.9 (PPML) |
| Phase 0 (PPML, employer panel) | Supporting; report PPML alongside the retained ratio | MDE about 2, region-year identified, positive non-IV signs suggest demand bias |
| FLS revision IV | Drop | No variation |
| 2. State minimum wages | Report as a minimum-wage result only; not an AEWR bound | Clean null for the floor; the AEWR translation shares the bite's gap model and needs λ = 1 and no scale effect |
| 3. Multi-region employers | Appendix mechanism check, or drop | Region-year identification, sign flip, lead problem |
| 4. Overtime laws | Drop as identification; keep the schedule | Lead failure, sign disagreement, 6 treated states, tiny dose |
| 5. Synthesis | Interim only. You judged the evidence inadequate on 2026-11-02; see `markdowns/identification_search_agent_plan.md` | One independent credible design (Track 1); the rest share the gap model or fail |

### Integration steps

Done this session:

- The Track 1 amendment registry and estimation script.
- The Track 4 schedule and script.
- The minimum-wage coverage producer fix: the code and a tracked source
  table.
- The README and report updates.

Still to do (each needs your go-ahead or an environment step):

1. **Rerun the Mundlak–Chamberlain branch** (`./scripts/run_mundlak_chamberlain.sh`)
   on the rebuilt panel. Its retained outputs still use the old farm floor.
   The minimum-wage artifact and the shared panel were already regenerated.
2. **FRED gap.** `code/utilities/pull_fred_minimum_wages.R` has no 2025/2026
   rows for AL, GA, LA, MS, SC, TN and WY (federal-floor states). That dropped
   7 states from Track 1's S5. Fix it at that producer by filling the federal
   minimum, a credentialed rerun.
3. **Track 1 as a design branch.** If you adopt it, add
   `code/designs/statewide_rule/` with a README, a page under
   `content/designs/`, and an assumption in `agent/assumptions.toml`. Rerun
   `08` unchanged when the FY2026 Q4 and FY2027 disclosures are released.
   They do not exist yet; the search plan works only with existing data.
4. **Overtime schedule.** If the welfare model uses it, promote
   `explorations/identification/sources/state_ag_overtime_schedule.csv` to a
   tracked a01 source table, as was done for the coverage table.
5. **PPML reporting and bounds framing.** Unchanged from before: add a PPML
   column to the panel-IV and plain-panel comparisons, and a bounds section
   in `markdowns/model_sketch.md`.

### Open questions

1. **Gate.** The declared primary fails the 1.0 gate (realized MDE 2.0). The
   secondary PPML row would pass (0.9). Promoting PPML to primary now would be
   a post-outcome choice and would have to be labeled as one. My
   recommendation: keep the declared primary and present PPML as the
   size-weighted secondary.
2. **Rebuild.** May I (or will you) regenerate `state_real_minwages.parquet`,
   the shared panel and the Mundlak–Chamberlain branch on the corrected
   coding?
3. **Framing.** Adopt the lower-bound plus welfare-sensitivity framing, with
   Track 1 as its anchor?
