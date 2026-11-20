# Plan: alternative identification strategies for the AEWR effect on H-2A use

Hand-off brief for an agent. Read it in full before starting.

## 1. Goal and the constraint every design so far has hit

The paper needs a credible, adequately powered estimate of how the Adverse
Effect Wage Rate (AEWR) affects H-2A use. Every design tried so far draws its
identifying variation from the AEWR moving region by region, year by year:
the plain panel regression, the panel IV, Mundlak–Chamberlain, the border
design and the bite interactions. That leaves about 220–240 region-year
deviations with an SD of roughly $0.40–0.46, against county outcomes of which
only 4–9% of the residual variance is shared within a region-year. With 17
clusters, these designs are underpowered whatever the estimator.

A design can escape this constraint only by doing one of the following:

1. **New policy variation within a region-year.** A rule that makes the
   AEWR differ across places that previously shared it.
2. **A different shifter of the same relative price.** Something that
   moves the cost of domestic labor, or of H-2A labor through a channel other
   than the AEWR, within a region-year.
3. **More signal per region-year.** Outcomes, aggregation or estimators that
   are far less noisy, so that the same variation goes further.
4. **A different paper.** Reframe around what can be bounded or ruled out.

This brief asks you to explore one track in each category. It is exploration.
Do not change any retained specification, table or pipeline stage. Deliver a
report and scratch scripts, and write any integration as a proposal for the
user to approve.

## 2. Read first

- `AGENTS.md`. Run `python scripts/agent_docs.py snapshot --scope code/designs`
  and read what it names, especially for `border_discontinuity`, `panel_iv`
  and `wage_reforms`.
- `markdowns/aewr_identification_notes.html` (Part 1 is the variance
  decomposition; Part 2 the panel IV algebra).
- `markdowns/bite_designs_exploration.md` and
  `markdowns/bite_designs_cross_design.md` (all bite results, the restricted
  wild bootstrap, and the MC unit mismatch).
- `markdowns/wage_reversal_2009_2010_exploration.md` and
  `outputs/reports/wage_reforms.md` (2009–10 reversal and herder phase-in).
- `content/designs/border-discontinuity.md` (section "Statewide-AEWR
  state-border design"), `code/designs/border_discontinuity/statewide_design.json`,
  `code/designs/border_discontinuity/13_statewide_power.R` and
  `outputs/reports/border_statewide_power.md`. These hold the 2025-rule
  registry and its power machinery, which Track 1 extends.
- `code/a01_sources/aewr_policy_events.csv` (dated rule events with sources).
- `markdowns/model_sketch.md` §1 (the welfare model takes the response of
  H-2A contract hours to the AEWR as its central behavioral input; this
  fixes which estimand matters).
- `explorations/bite/README.md` (script conventions, the hand-rolled
  bootstrap, and `cross_design_helpers.R`, which you should reuse).

## 3. What is already known (do not redo)

| Design | Result | Why it fails or stalls |
| --- | --- | --- |
| Plain panel OLS, county + year FE | certified share −0.044 (0.039), region CRV1; MDE ≈ 119% of the mean | Region-year variation only; a lead is ~60% of the current coefficient |
| Panel IV (column 4) | first stage 0.170, F 44.7; IV −0.089 (0.063); MDE ≈ 190% of the mean | Exclusion defensible (dissimilar donors), precision is not |
| Bite 1(b), frozen ACS gap × AEWR | 0.0015 (0.0012); MDE ≈ 4% of the mean | Identifies heterogeneity by bite, not the average effect; hours fail the pre-test |
| Bite 1(a), adds E × year FE | MDE ≈ 50% of the mean | Back to region-year variation |
| Border (v6 rings), bite interaction | MDE ≈ 55–70% of the mean; the lead also predicts the outcome | 25 edges among 16 regions |
| Census pay exposure; employer bindingness | No relevance; not robust across cohorts | See bite report §1 and §4b |
| 2009–10 rule reversal | Clear offered-wage response; county response looks like a trend | Needs the 2009 local wage schedule archive (not held locally); pre-period starts 2008 |
| 2023 occupation-specific AEWR | Not pursued | Affects too few positions (user decision) |
| Herder phase-in 2015–18 | Large wage first stage | No convincing comparison group; participation response precedes the wage change |
| Statewide 2025 rule, **border** version | Design-stage power: 0 of 64 combinations pass | Size gates fail with 22 within-region pairs |
| FLS preliminary-vs-revised revision IV | Revisions are essentially zero | See below |

**FLS revision instrument (checked while writing this brief):** in
`data/intermediate/fls_region.parquet`, the November preliminary
field-and-livestock wage equals the next year's revised value in 96% of
391 region-years (2001–2023). Over 2010–2022, the within-year SD of the
revision is $0.0007. There is no variation to instrument with. Reproduce
this in a short script and confirm two things: that the parser reads the
preliminary and revised values from different annual tables, and that the
published AEWR equals the preliminary value. If both hold, list the design
as dead and move on.

## 4. Phase 0: common unit, support and precision diagnostics (do first)

### 4a. One unit for every design

Report every headline estimate in two units:

- per real dollar of AEWR (or of the relevant price), as before;
- as an elasticity at the estimation-sample means:
  β × mean(price) / mean(outcome).

Also convert the existing headline estimates in §3 into elasticities, so the
user can compare all designs on one scale. The welfare model needs a contract
hours response, so report certified hours alongside certified positions
wherever the data allow.

**Proposed gate (the user may change it):** a design is "promising" when the
elasticity MDE (2.8 × SE × mean price / mean outcome) is at most 1.0. In
addition, it must pass the pre-trend and lead placebos, survive the
appropriate wild bootstrap, and survive at least one alternative
treatment/exposure measure. Also report the earlier gate (MDE below 25% of the
outcome mean per real dollar) so the results can be compared with the bite
report.

### 4b. More signal per region-year (category 3)

On the plain panel regression and the panel IV, keep the treatment and
identifying variation fixed. Ask only how much precision changes with the
outcome and estimator. Specifications:

1. Baseline: the retained county ratio outcome, equal county weights.
2. Poisson pseudo-maximum likelihood (`fepois`) on certified-position and
   certified-hour counts with county and year FE. This estimates a
   semi-elasticity of total H-2A use and weights by size. Report it with the
   same clustering.
3. The same ratio outcome weighted by 2011 farm employment.
4. The region-year aggregate: log certified positions (and hours) by
   region-year, region and year FE, 17 clusters.
5. An employer panel for employers present at baseline (use the
   `employer_id_balanced` cohort machinery in `explorations/bite/employer_checks.R`),
   with outcome log1p(positions) and FE for employer and year. Respect the
   2020 linkage break: run 2011–2019 separately.

For each, report the estimate, its SE, the elasticity MDE, and the share of
residual outcome variance at the region-year level. That share is the
quantity that determines whether a region-year design can ever be powered.
Precision gains from PPML or weighting are worth having, but they don't touch
exclusion. Say so, and state that weighting changes the estimand.

### 4c. Treatment support for the new tracks

Before estimating anything in Tracks 1–3, tabulate the identifying variation
each one would use, after its own fixed effects:

- **Track 1:** the within-former-region SD of the 2025 statewide rates across
  states. Single-state regions (check which: California, Florida and Hawaii
  are candidates) contribute no within-region contrast and drop out.
- **Track 2:** the number of state-years with a minimum-wage change, the
  within-region-year SD of the real applicable agricultural minimum wage, and
  the share of counties whose frozen ACS p10/p25 lies within $1 of it.
- **Track 3:** the number of employers that place workers in two or more AEWR
  regions in the same year, their share of positions, and how this changes over
  time.

If a track's support is negligible, stop that track and report why.

## 5. Tracks

### Track 1 (category 1, highest priority): the 2025 statewide rule, all counties

**Why:**

- The interim final rule effective 2025-10-02 replaced regional FLS AEWRs
  with statewide OEWS-based rates by skill level. States that shared one AEWR
  through FY2025 have different ones from FY2026.
- This is the only large policy change found so far that varies within former
  regions, and it applies to all field and livestock positions.
- The border version fails on power because it uses only 322 counties in 22
  pairs. An all-county version uses every county in a multi-state former
  region, with states (about 40+) as clusters instead of 17 regions or 22
  pairs.

**Data:**

- Statewide rates: `data/intermediate/aewr_occupation_skill_rates.parquet`.
- Regional pre-period rates: `aewr_rate_schedule.parquet`.
- Case records through FY2026 Q3: `h2a_disclosure_records.parquet`, with
  20,638 FY2026 records, against 24,526 for all of FY2025.
- The shared `h2a_data_year.parquet` stops at 2025, so build the FY2026
  outcome from the case records, following the owning producer's allocation
  logic. Do not write a parallel allocation.

**Specification to declare:**

- Unit: county × federal fiscal year, FY2012–FY2026.
- Treatment: the rate in effect on April 1 of each FY. Before FY2026 that is
  the regional FLS reference; from FY2026 it is the statewide U.S.-worker
  skill-I rate. This matches `statewide_design.json`. Sensitivities are the
  H-2A skill-I cash rate after the housing adjustment, and U.S. skill II.
- Fixed effects: county, and former-AEWR-region × FY. With those FE, the only
  variation left is the post-2025 within-region dispersion. The national level
  change is absorbed and not identified, so say so.
- **Comparable seasons:** FY2026 is observed only through Q3. Build every FY's
  outcome from decisions in Q1–Q3 only (October–June), so pre and post
  periods cover the same seasonal window. Report a full-year pre-period version
  as a check on how much Q4 matters.
- Cohort timing: job orders submitted before 2025-10-02 carry the old rate.
  Classify cases by the date that determines applicability, using the
  `wage_reforms` and border-branch conventions, and read the rule text in the
  event file for the applicability trigger. The main estimate is
  intent-to-treat on the statutory rate. A cohort-aligned version is a
  sensitivity.
- Outcomes: certified positions and certified hours per fixed 2011 farm
  worker, and the PPML counts from Phase 0.
- First stage: hours-weighted offered hourly wage on the statewide rate, the
  same pass-through check as the border registry.
- Event study: dose × FY, where dose is the FY2026 statewide rate minus the
  FY2025 regional rate. Use FY2012–FY2025 as pre-years. Fourteen pre-years is
  the strongest pre-trend test available anywhere in this project.
- Inference: CRV1 by state; a restricted Webb wild bootstrap over states; and
  randomization inference permuting dose across states within former region.

**Order and stop point (research integrity):**

1. Do Phase 0 support and the design-stage power simulation first. Reuse the
   DGPs, seed and size gate in `13_statewide_power.R`, with all counties
   instead of border pairs. Build the pre-period residual scale from pre-FY2026
   data only. **Do not parse or inspect any FY2026 outcome, offered wage or
   case count before this step is finished.**
2. Write a registry JSON next to your scripts, in the format of
   `statewide_design.json`. It declares the specification, primary and
   sensitivity choices, and the gate. Record its SHA-256 in the report.
3. **Stop and report to the user** with the power results and the
   registry. Read post-policy outcomes only after the user approves, unless
   the hand-off message explicitly pre-authorizes continuing.
4. After approval, estimate exactly the declared specifications. Report all of
   them.

**Threats to test or state:**

- **The dose comes from state OEWS wages.** States with high farm wages
  relative to their former region get a higher AEWR. Test with the long
  pre-period. Add state-specific linear trends as a sensitivity, and the real
  state minimum wage as a control.
- **Concurrent 2025 shocks that differ by state**, above all immigration
  enforcement intensity, which may raise H-2A demand. Nothing in the repo
  measures it. Name it as the main unaddressed threat. Check whether dose
  correlates with FY2025 H-2A growth, which would be a warning sign.
- **Legal status.** You cannot verify current litigation or any stay of the
  interim rule without a network call. Ask the user. Do not assume the rates
  were applied throughout FY2026.
- **Skill level is an employer choice.** Keep the statutory skill-I rate as
  the intent-to-treat treatment.

### Track 2 (category 2): domestic labor cost via state minimum wages

**Why:**

- The H-2A decision depends on the AEWR relative to what domestic labor costs.
  State minimum wages move the domestic floor across states within the same
  region-year, while the AEWR stays fixed.
- Clusters are states, not regions.
- The data are in the shared panel: `state_min_wage`, `prevailing_ag_min_wage`
  and their real versions.

**Specifications:**

1. y ~ real applicable agricultural minimum wage | county + region × year.
   Cluster by state.
2. Bite triple difference: the minimum wage should matter only where it binds
   on domestic workers. Interact it with a frozen measure of how close local
   low wages are to the floor. Use the 2008/2010 ACS p10 and p25 from
   `explorations/bite/cross_design_exposure.R`, frozen and not
   contemporaneous.
3. Contiguous cross-state county pairs **within the same former AEWR region**,
   with pair × year FE (Dube–Lester–Reich). Reuse the statewide border
   geography builder (`statewide_design.json`, spec
   `statewide_h0_unscreened_within_region`) rather than writing a new one.
   These pairs share an AEWR through FY2025, so any outcome difference over
   2011–2025 comes from domestic floors. This also gives the border branch
   pre-2025 contrasts it has never had.
4. **Relative-price restriction test:** without region × year FE, estimate y
   on the real AEWR and the real minimum wage jointly. Test whether the
   coefficients are equal and opposite, as they would be if only the gap
   between them matters. Use this to state how far a minimum-wage response can
   be translated into an AEWR response, and under what assumption.

**Threats:**

- Minimum-wage increases follow state economic conditions and political
  trends. Event studies around large increases, with state-pair FE, are the
  main defense.
- A minimum wage also changes farm scale and output directly, not only
  substitution between H-2A and domestic workers.
- **Agricultural exemptions.** The bite report found that the minimum-wage
  producer applies the 2024 agricultural-exemption classification to every
  year. Audit this first: list the states whose agricultural coverage changed
  over 2008–2025 according to the producer's source notes. If the
  classification is wrong for years that carry identifying variation, report
  that, and propose a producer fix rather than repairing it downstream.

### Track 3 (categories 2 and 3, cheap): multi-region employers

Employers that place workers in several AEWR regions in the same year (custom
harvesters, beekeepers, farm labor contractors) face different AEWRs at the
same time.

- Specification: the share of the employer's positions or hours in region r
  in year t, regressed on the real AEWR_rt, with FE for employer × year and
  employer × region.
- What it buys: employer-level demand shocks and composition are removed.
- What it doesn't fix: identification is still region-year, and regional
  demand shocks (crop conditions) remain. Add region-level crop price and
  yield controls if they exist in the shared panel.
- Inference: two-way clustering by region and employer, plus a region wild
  bootstrap.
- Expect limited power. Run this only if Phase 0 support shows enough
  multi-region placements, and report it as a mechanism check.

### Track 4 (category 2, specification only): state farmworker overtime laws

Several states phased in farmworker overtime over 2019–2025. H-2A workers are
covered by state law, so overtime raises H-2A cost per contract. The
increase is larger for jobs with long weekly hours. A job-level dose:

- dose = 0.5 × max(0, weekly hours − state threshold_t) / weekly hours;
- weekly hours from `number_of_hours` on each job order;
- with the baseline hours mix frozen at the employer or county level.

This gives a labor-cost shock with variation within a state-year. The
estimand differs from the AEWR effect, because domestic costs at the same
farms rise too. Still, it yields an H-2A hours response to an hourly cost
increase, which is close to what the welfare model needs.

**Do not estimate this track.** The repo has no state schedule of overtime
thresholds, and compiling one means source acquisition. Write the specification,
the data needed (state, effective dates, thresholds by employer size), the
support check you would run, and the inference plan. Few treated states means
you need few-treated-cluster methods (MacKinnon–Webb; Conley–Taber). Ask
the user whether to acquire the schedule.

### Track 5 (category 4, always do): evidence synthesis

Whatever Tracks 1–4 show, build one table of what each design can rule out:

- **Rows:** every design and specification, including the earlier failures.
- **Columns:** point estimate, the 95% interval in elasticity units (use
  Anderson–Rubin sets for IV, and the wild bootstrap where available), and the
  estimand (average effect, differential effect, relative-price elasticity,
  ITT on a 2025 cut).
- **Then:** state which elasticities each design excludes, and whether any
  combination of designs, each under its own stated assumptions, excludes
  large responses.

This feeds a fallback framing of the paper: bounds plus welfare sensitivity
over the range not excluded (see `markdowns/model_sketch.md`). Do not pool
estimands that differ. Show them side by side.

## 6. Inference and reporting rules (all tracks)

- Use the cluster level of the identifying variation: states for Tracks 1–2,
  regions for region-year designs, and pairs plus states for border pairs.
  Report the alternatives in the full CSV.
- Use a restricted Webb wild cluster bootstrap for every headline coefficient.
  Reuse `cross_design_helpers.R`; `fwildclusterboot::boottest` fails on these
  models.
- Report the MDE (2.8 × SE) in outcome units, as a percentage of the mean, and
  as an elasticity.
- Report every specification run, including failures. Do not select on
  significance. If a choice is made after seeing outcomes, label it as such.

## 7. Known data pitfalls

- The 2009 AEWR is missing; 2009–2010 were under the 2008 rule.
- From 2023, the AEWR is occupation-specific for non-field occupations, and
  `cert_hours_at_aewr` is zero from 2023.
- Employer counts break in 2020 (about 10k to 36k employers) when
  association and contractor filings start naming member businesses.
- County H-2A counts are fractional where multi-county cases are allocated.
- ACS quantiles are commuting-zone level and real in the shared panel, but
  nominal in `data/intermediate/acs_czone_wage_quantiles.parquet`.
- The Mundlak–Chamberlain fraction-affected mixes nominal and real units.
  Do not compare against it without saying so.
- BEA controls end in 2023. Use the continuing-controls set for 2024–2025 and
  check what exists for FY2026.
- FY2026 disclosures are partial (through Q3) and a single vintage, so they
  may be revised.

## 8. Constraints

- No commits.
- No source acquisition, network calls or credentialed steps. Stop and ask
  where a track needs one (Track 1 legal status, Track 4 schedules).
- Don't edit retained outputs, `data/raw`, crosswalks or producers. Propose
  producer fixes in the report.
- New scripts go in `explorations/identification/`, with a README in the style
  of `explorations/bite/README.md`. Numerical outputs go to a scratch
  directory outside the repo, given as each script's first argument.
- Geographic IDs stay strings. Use `code/c00_shared/geography.R`.
- R: `R_LIBS_USER=$(ls -d $PWD/renv/library/*/R-*/* | head -1) Rscript --vanilla <script> <out-dir>`,
  then `git checkout -- Rplots.pdf`. Python:
  `UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync`.
- Any suggested specification change is a proposal naming the README or design
  page to update. Do not implement it.

## 9. Deliverables and order

1. Phase 0 (§4) and the FLS revision confirmation.
2. Track 1, up to and including the stop point; then Tracks 2, 3 and 5
   while waiting for approval; Track 4 as a written specification.
3. Track 1 post-policy estimation, only after approval.

Report: `markdowns/alternative_identification_exploration.md`, with:

1. a one-paragraph verdict at the top;
2. the common-unit table converting earlier results to elasticities;
3. the Phase 0 precision and support diagnostics;
4. one section per track, giving specification, support, estimates, MDEs,
   pre-trends, placebos and bootstrap p-values, or the reason it stopped;
5. the synthesis table of what is ruled out;
6. a recommendation for each track (main design, supporting evidence or drop),
   proposed integration steps, and the questions that need the user's
   decision.

Every number in the report must trace to a script named in it.
