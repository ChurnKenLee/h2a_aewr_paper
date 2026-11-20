# Identification search: agent plan

**Status:**
- Written 2026-11-02 after the user judged the findings in
  `markdowns/alternative_identification_exploration.md` inadequate.
- Updated the same day with the user's decisions (below).
- Updated again: run as a single **sequential** workflow (one agent at a
  time) of Claude agents, with blind review by fresh agents (user, 2026-11-02).
- Nothing here has been run.

**Goal:** an estimate of how H-2A use responds to the AEWR (more precisely,
to the hourly cost of H-2A labor) that is **credible and precise enough to
tell a conventional elasticity (about −0.3) apart from both zero and −1.**
That needs a size-valid elasticity MDE of about 0.5 or less for a single
design, or about 0.3 for a pre-declared pooled estimate.

**The user needs results now.** Only data that already exist can be used:
FY2008 through FY2026 Q3, plus any public historical source that can be
acquired. FY2026 Q4 and FY2027 disclosures do not exist yet. Nothing in this
plan waits for them.

## Decisions recorded (2026-11-02)

| # | Decision | Status |
| --- | --- | --- |
| D1 | Outcome | **Hours** (user). Primary: total certified H-2A contract hours, size-weighted (PPML on counts, or an aggregate elasticity), as the response to H-2A cost per hour. **Requested positions are the lead secondary;** see "Outcomes" for why they are not primary. |
| D2 | Gate | Not answered, so the recommended default applies: single-strategy size-valid MDE ≤ 0.5 is "informative", ≤ 1.0 is pooling-eligible, and the pooled target is ≤ 0.3. The user may override before Phase 3. |
| D3 | Data | **Full network and write access** (user). All Phase 2 items are pre-approved. See "Repository safety" for how write access is interpreted. |
| D4 | Reserved data | **Moot.** FY2026 Q4 and FY2027 do not exist. The integrity rules below replace data reservation. |
| D5 | Pooling | Not answered, so the recommended default applies: a pre-declared pooled estimate across strategies with different assumptions may be the headline. |
| D6 | Scale and execution | **Run sequentially from the main session: one subagent at a time** (user, 2026-11-05). No parallel workflow. Every agent is a Claude agent. Independence comes from fresh agents that see only files, not from a second model. Roughly 30–45 agent runs in total; wall-clock time is the sum of the runs (many hours). |

## Outcomes

Data facts, checked 2026-11-02:

- **Requested positions** (`nbr_workers_requested`) exist in the all-case
  disclosure records, which include denied and withdrawn cases, for
  FY2011–12 and FY2015–FY2026 Q3.
  - **Missing entirely in FY2008–10 and FY2013–14.**
    `nbr_workers_needed` may fill some of these years; Phase 0 must check.
  - Certified positions are 93–98% of requested in those records.
  - In the allocated worksite file and the county panel, requested ≈
    certified (ratio 0.997–1.000), so that version adds almost nothing.
  - Any requested measure must be built from the all-case records.
- **Certified hours** exist in every year, FY2008–FY2026 Q3. Hours equal
  positions × contract weeks × weekly hours. Weekly hours are missing in
  about 0.6–0.7% of rows, which are imputed.

| Role | Outcome | Why |
| --- | --- | --- |
| **Primary** | Total certified contract hours (size-weighted) | What the welfare model needs (`markdowns/model_sketch.md`); available in every year, including 2009 (A2) |
| **Lead secondary** | Requested positions from all cases, including denied and withdrawn, dated by **receipt** date | Closest to the employer's demand decision; not filtered by DOL review. Receipt dating avoids processing delays such as the October–November 2025 shutdown. Unavailable for FY2008–10 and FY2013–14 unless `nbr_workers_needed` fills them. |
| Secondary | Requested hours (requested positions × requested dates × weekly hours); certified positions | Shows whether the response runs through headcount or contract length |
| Secondary | Equal-county ratio outcomes (the earlier primary) | Continuity with the earlier reports |
| Secondary | Extensive margin: any use, employer entry and exit | Where the alternative wage matters |
| Decomposition | log hours = log positions + log weeks + log weekly hours | Which margin responds |

**Why requested positions are not the primary:** they are missing in the
2009 reversal years and in FY2013–14. They track certified positions within
2–7%. And they ignore the duration and weekly-hours margins that the welfare
model prices.

**When requested positions are primary within a strategy:** in any strategy
confined to FY2015+ (for example A1 and A3), the registry may make them
co-primary with hours. It must then use a multiplicity adjustment.

## Where things stand

Read before planning anything. Do not re-run these designs except as
declared inputs to pooling.

| Evidence | Status | Source |
| --- | --- | --- |
| Track 1: 2025 statewide rule, FY2025 vs FY2026 Q1–Q3 | **The only credible design.** First stage 0.66 (offered wage per statutory dollar), clean placebo. No detectable quantity response: certified positions elasticity 0.40 [−1.06, 1.86], size-weighted PPML −0.12 [−0.77, 0.53]. MDE 2.0 (equal-county), 0.9 (PPML). One season. **Certified FY2026 Q1–Q3 outcomes have been seen;** requested positions and hours have not been examined. | exploration report §3; `explorations/identification/08_track1_estimate.R`; both Track 1 registries |
| Region-year AEWR designs (plain panel, panel IV, PPML, employer panel, cluster IVs, FLS revisions) | **Capped.** Region-year cells hold 1.4–5.7% of residual variance across 17 clusters. MDEs 1.5–25. | report §1–2; `improved_dissimilarity_cluster_iv.md`; `aewr_measurement_error_instrument.md` |
| Gap-model designs (bite 1(a)/1(b), Mundlak–Chamberlain bite coordinates, Track 2 minimum wages) | **Share one untestable model.** No domestic-wage measure predicts binding: 94% of 2011–13 hourly H-2A hours sit at the AEWR, within-region R² < 0.002. The Track 2 translation also needs λ = 1 and no scale effect. | report §4, §7; `bite_designs_exploration.md`; `predetermined_aewr_bite_design.md` |
| Border designs (v6 ring, statewide border power) | MDE 4–7; the lead fails for positions | report §1 |
| 2009–10 wage-method reversal | Wages moved (58% of 2009 cases offered below FLS) and the dose varies within regions. But the county response looks like a trend; there is only one pre-year; legal uncertainty blurs timing. **Two blockers:** verified FLC Level I schedules, and FY2006–07 disclosures. | `wage_reversal_2009_2010_exploration.md`; `code/designs/wage_reforms/`; `outputs/reports/wage_reforms.md` |
| 2023 occupation-method rule | Designed, not estimated. About 3% of positions affected. | `aewr_2023_rule_change_design.md` |
| Overtime laws (Track 4), multi-region employers (Track 3), FLS revisions | Not credible or no variation | report §2a, §5, §6 |

**Lesson for triage.** Precision depends on three things:
1. the dispersion of the identifying dose after fixed effects;
2. the number of independent units carrying that dose;
3. outcome noise at the unit of identification.

Track 1's dose is large (a $4 cut), but its *within-region* dispersion is
only $1.09 across 46 states. Every candidate must report all three numbers
before anything else is done.

## Rules for every agent

**Integrity.** No fresh data will arrive. These rules replace "read new data
only after freezing":

1. **Freeze before estimating.** For each strategy, write a registry (JSON,
   SHA-256 recorded in the ledger) before computing any estimate for it.
   - Allowed before freezing: treatment variables, support statistics, and
     pre-period outcome *residual variance* for power calibration.
   - After freezing, run only what the registry declares. Anything else is
     labeled post-outcome.
2. **Confirmation split, where tuning is needed.** If a strategy needs
   data-driven choices (bins, donor sets, classifiers, bandwidths), the lead
   assigns units with seed 20261102 into two parts:
   - an *exploration* third, used to make the choices;
   - a *confirmation* two-thirds, used once with the frozen registry.

   Units are states or sub-state units, whichever the strategy has enough
   of. Strategies with no tuning skip the split. Strategies with fewer than
   about 30 independent units skip it and are flagged.
3. **Prior exposure.** Agents and earlier reports have already looked at
   most historical data. Each registry states which of its outcomes and
   periods earlier work has examined, citing the report section.
4. **No selection on results.**
   - Every strategy that reaches estimation is reported, and every strategy
     killed earlier is logged with its reason.
   - Report Romano–Wolf adjusted p-values across all estimated primary
     hypotheses; Holm if Romano–Wolf is infeasible.
   - The pooled estimate uses the inclusion rule fixed in Phase 0, based on
     design gates, never on estimates.
5. **Blind review.** The agent that implements a strategy does not review
   it.
   - Each review is a fresh Claude agent that did not take part in that
     strategy.
   - Reviewers get the registry, code and output files, not the
     implementer's conclusions, and ignore interpretive prose in them.
   - (Codex cross-review is not available from the workflow. The user may
     add it later by running Codex on the same files.)
6. **Units.** Every estimate is converted to the primary estimand: the
   elasticity of hours with respect to H-2A cost per hour. Show the
   conversion and its assumptions.

**Repository safety.** These follow `AGENTS.md`. The user granted full
network and write access, interpreted as follows:

- **Network:** any public source may be acquired without asking. Never
  print or copy credentials. Credentialed APIs use only keys already in
  `.env`.
- **Writing:**
  - Agents may create scripts under
    `explorations/identification_search/<strategy_id>/`.
  - They may create new sourced tables (citation, URL, retrieval date and
    confidence per row, modeled on
    `code/a01_sources/state_ag_minwage_coverage.csv`).
  - They may put raw downloads under `data/raw/<new_source>/`, and add new
    a01/b01 producers and intermediates for newly acquired sources, with
    README entries.
- **Still not allowed without asking:**
  - changing an existing producer's output schema or values;
  - editing retained outputs or the manuscript;
  - running `scripts/run_all.sh`;
  - committing;
  - deleting anything.

  Analysis outputs go to a scratch directory outside the repository.
- Geographic IDs stay strings (`code/c00_shared/geography.R`, `h2a.geography`).
- R: `R_LIBS_USER=$(ls -d $PWD/renv/library/*/R-*/* | head -1) Rscript --vanilla <script> <out-dir>`,
  then `git checkout -- Rplots.pdf`. Python: `uv run --no-sync`.

**Stop and ask the user** only when:
- an action would change existing producers or retained results;
- two agents disagree on a kill or a review verdict after one exchange;
- the evidence suggests the primary estimand itself is wrong.

## Phase 0: estimand, units and shared tools

One lead agent; target under one session.

1. **Outcome build.** One script that builds the Outcomes table for every
   county × FY and employer × FY:
   - Certified hours use the b01 fiscal-year logic, validated as in
     `04_track1_power.R`.
   - Requested positions and requested hours come from all-case
     disclosure records, dated by receipt.
   - Check `nbr_workers_needed` coverage for FY2008–10 and FY2013–14 and
     report whether it can stand in for requested positions.
   - Also build a calendar-year (start-year) version.
2. **Common cost unit.**
   - Build an **H-2A cost per hour** series for each county-year and job
     order: the binding statutory wage floor (AEWR, prevailing, minimum or
     CBA, where known), plus per-hour housing, transportation and fees.
     Declare imputations.
   - Shifters of different cost components then map to one unit.
   - "Cost equivalence" (employers respond to total cost per hour,
     whatever its component) is an assumption. Test it where two components
     vary.
3. **Shared power harness.** Generalize `explorations/identification/04_track1_power.R`
   into a function. Inputs: any unit × period dose matrix, fixed-effect
   structure and cluster structure. Outputs:
   - size-valid MDE, using the jackknife and randomization inference
     wherever CRV1 fails the size gate;
   - the effective number of clusters;
   - the maximum leverage share.

   Keep the border branch's error processes and gate: seed 201103, 999
   repetitions, AR(1) 0.5, upper Wilson bound ≤ 0.075.
4. **Strategy ledger.** `explorations/identification_search/ledger.csv`,
   one row per strategy. Columns:
   - strategy_id, family, proposer and implementer (model);
   - estimand and conversion assumptions;
   - support statistics;
   - design-stage MDE;
   - registry path and SHA-256;
   - whether a split is used;
   - phase reached, kill reason, reviewer, review verdict.
5. **Pooling rule** (declare now). Pool only strategies that:
   - pass their own size and falsification gates;
   - convert to the primary estimand;
   - rest on different identifying assumptions. Gap-model designs count
     as one.

   Use random-effects inverse-variance pooling and report heterogeneity
   (Q, I²). Report each strategy alone first.
6. **Record D1–D6** at the top of the ledger.

## Phase 1: triage

Two triage agents run one after the other, each with a different emphasis:
one weights credibility (exogeneity and falsification), the other power
(support, effective clusters, MDE). The second must not read the first's
output. Each scores the whole catalogue below and adds at least three
strategies not on it. A merge agent then combines them.

**Triage uses no outcomes.** It uses treatment data, support statistics, and
pre-period residual variance for a rough MDE.

**Rubric** (score 1–5 each; record in the ledger):

| Criterion | Question |
| --- | --- |
| Exogeneity | Why is the dose unrelated to H-2A demand shocks, and what falsification test exists? |
| Estimand distance | How many assumptions convert it to the primary elasticity? |
| Support | Dose SD after FE × number of independent units; the rough MDE |
| Independence | Does it rest on an assumption no other shortlisted strategy uses? |
| Speed | Can it produce an estimate within this wave (local data, or one quick public download)? |

**Kill at triage** if the rough MDE exceeds 2 and the strategy cannot be
pooled, or if the dose dispersion is region-year only (17 clusters) with no
added cross-sectional structure.

**No user checkpoint:** the lead advances every strategy that survives
triage, up to the agent cap, ordered by rubric score.

### Catalogue (seed list; agents add to it)

**A. Statutory formula shocks to the AEWR itself.** Highest priority: the
dose is the AEWR, so no translation is needed.

- **A1. 2025 statewide rule, extended with the data that exist** (FY2025 vs
  FY2026 Q1–Q3).
  - **New outcomes** (Outcomes table):
    - hours (the new primary);
    - requested positions by receipt date, not yet examined for FY2026;
    - extensive margin;
    - employer-level changes for existing FY2025 employers.
  - **Occupation dimension:** the `Statewide 655.120(b)(1)(ii)` rates for
    non-field SOCs are already in `aewr_occupation_skill_rates.parquet`.
    They give within-state variation across occupations.
  - **Housing-adjustment dimension:** the gap between H-2A and U.S. rates
    varies by state.
  - **Revealed binding (D1 below):** interact the dose with each unit's
    FY2025 share of hours at the AEWR.
  - **Prior exposure:** certified positions, certified hours and equal-county
    ratios for FY2026 Q1–Q3 have been seen (exploration §3). New registries
    must say so.
  - **Not available:** the 2026-08-03 schedule's season (FY2026 Q4 and
    later). Exclude it.
- **A2. 2009 OES switch and 2010 return to FLS.**
  - **Dose:** set by OES area, occupation and skill level, so there are
    hundreds of sub-state units with a large dose (offered wages 12% below
    FLS).
  - **Acquire first (pre-approved):**
    - the 2009 OES-based AEWR schedules by area, occupation and skill
      (archived FLC Data Center tables; Wayback Machine);
    - FY2006–07 H-2A disclosures (OFLC performance data archives).
  - **Rework the earlier exploration's trend problem:**
    - use state × year fixed effects with the sub-state dose;
    - use the 2010 reversal as a symmetric second event;
    - run case-level timing around 2009-01-17 and 2010-03-15.
  - **Outcome:** certified hours (requested positions are missing in
    FY2008–10).
  - **Threat:** the 2009 recession differs across counties. Control with
    pre-period construction and manufacturing exposure, declared in the
    registry.
- **A3. 2023 occupation-method rule** (`aewr_2023_rule_change_design.md`).
  - **Treated jobs:** non-field SOC jobs moved to statewide OEWS rates on
    2023-03-30.
  - **Design:** a monthly employer × job-cell panel; FY2021–FY2026 Q3 data
    exist.
  - **Small share (about 3%),** but the doses are large.
  - **Acquire:** the 2023–24 OEWS occupation AEWRs (DOL; pre-approved).
  - **Model the Kansas injunction** (2024; entity-level) explicitly.
- **A4. Range herder phase-in, 2015–2018** (`code/designs/wage_reforms/`).
  - **Shock:** a national monthly floor of $1,206.31 from 2015-11-16,
    $1,389.67 in 2017 and $1,584.22 in 2018 (`content/designs/wage-reforms.md`).
    The pre-rule level needs a source.
  - **Dose:** varies by state through the pre-rule wage level and higher
    state floors.
  - **Estimand:** a different occupation, but a real wage-floor elasticity.
    Use for pooling only with an explicit external-validity flag.
- **A5. Annual-update timing shocks.**
  - **Events:** the delayed 2021 update (2021-02-23), and the 2024 Kansas
    injunction split (2024-12-16 vs 2024-12-30).
  - **Expected value is low** (doses of weeks). Run triage only, unless an
    agent finds a sharper margin.
- **A6. Stacked statutory events.**
  - **Design:** stack A1–A4 in one design, with event-specific fixed
    effects and doses converted to the common cost unit.
  - **Declare the stacking rule** before any per-event estimate is read.
  - **Results:** run once the per-event registries are frozen.

**B. FLS measurement instruments.** Region-year, so they are capped. Pursue
only with added structure.

- **B1. FLS sampling-composition shocks** (`fls_realized_sampling_weight_instrument.md`,
  `fls_sampling_weight_*`, `fls_realized_geography_entropy_balancing.md`).
  - **First,** report the share of AEWR variance the recovered noise
    explains, within region.
- **B2. AEWR minus an independent contemporaneous regional wage.** Compare
  against QCEW crop pay, OEWS and ACS farm wages; the residual should be
  mostly survey noise. First test whether that residual predicts any
  pre-period outcome trend.
- **B3. Within-region composition shift-share.**
  - **Idea:** the regional AEWR moves when other states' share of the
    region's farm employment shifts (QCEW), with no change in the focal
    state's own wages.
  - **Approach:** build a leave-own-state-out version.
- **Kill rule for B:** if the instrument's within-region-year first-stage
  F is below 10 with region clustering, or the design-stage MDE exceeds 2,
  stop.

**C. Non-wage H-2A cost shifters.** These give the *own-price* response
through cost equivalence, unlike Track 2's cross-price effect.

- **C1. Housing.**
  - **Cost:** H-2A employers must house workers. Per-hour housing cost =
    county rent level (HUD Fair Market Rents, public, county-year) ÷ (beds
    per unit × contract hours).
  - **Dose:** housing-cost changes driven by non-farm housing demand.
    Candidate instruments:
    - the national rent cycle × local housing-supply elasticity;
    - metro rent spillovers into nonmetro counties.
  - **Threat:** rents track local labor-market tightness. Test with
    domestic farm wages as an outcome.
  - **First triage number:** dose size in dollars per hour. A rough
    estimate is a few cents a year, which may be too small.
- **C2. Inbound and outbound transportation.**
  - **Cost:** employers reimburse travel. Per-hour transport cost =
    distance from the sending region or consulate × fuel or fare prices
    (EIA diesel, public) ÷ contract hours.
  - **Dose:** a shift-share of national fuel shocks × fixed county
    distance.
  - **Threat:** diesel also raises other farm costs. Control with diesel
    intensity by crop mix × fuel price.
- **C3. Visa and consular processing shocks.**
  - **Events:** 2020 consular suspension, interview waivers, fee changes.
  - **Exposure:** counties weighted by pre-period sourcing shares.
  - **First check:** does any data identify worker origin or consulate? If
    not, kill.
- **C4. Mandated meal and subsistence rates:** national only. Triage and
  likely kill.

**D. Revealed bindingness** (replaces survey-based bite).

- **D1. Binding share × statutory change.** Instrument the change in offered
  wage with (statutory change) × (pre-period share of hours offered at the
  AEWR).
  - Works with A1, A2 and A3.
  - Employers already paying above the new floor get a smaller effective
    dose. Under the 2025 cut that is visible directly: the first stage is
    0.66, not 1.
  - Uses observed behavior, not a survey wage.
- **D2. Jobs where another floor binds as controls.**
  - **Idea:** where a state prevailing-wage finding (piece or hourly) or a
    CBA exceeds the AEWR, AEWR changes should not matter. Use those jobs as
    a placebo or control group.
  - **Acquire:** SWA prevailing-wage findings (ETA-232 / AOWL / FLAG;
    pre-approved). If coverage is too thin, kill quickly.

**E. Outcome and margin redesign.** Layered onto A, C and D as declared
secondaries, not standalone strategies.

- **E1.** Job-order or monthly panels of applications and requested
  positions (closer to demand; high frequency).
- **E2.** Extensive margin: employer entry and exit, and any use in a
  county.
- **E3.** Stable employer × worksite × job-type cells (as in the 2023
  design).
- **E4.** Aggregate state or region totals (the welfare estimand).
- **E5.** Variance reduction: regression adjustment on pre-period trends,
  and synthetic-control or SDID weights for A1, declared before outcomes.

**F. Structural and revealed-preference.** Model-based; labeled as such.

- **F1. Counterfactual-wage recovery.**
  - **Data:** FY2026 offers above the new, lower floor (and pre-2009
    above-AEWR offers) reveal employers' market wage.
  - **Use:** estimate the distribution of the alternative wage. This is the
    quantity the gap designs could never measure.
- **F2. Adoption model.**
  - **Model:** a discrete choice of H-2A use with a fixed cost and
    per-hour cost, estimated on A-family policy variation.
  - **Validation:** with no future data, hold out one A-family event (for
    example A2), declared before fitting, and predict it before
    estimating it.

**G. Geographic contrasts.** Already tried (border, statewide border,
Track 3). Revisit only if an agent finds new structure, for example a
sub-state dose from A2 at OES-area borders.

## Phase 2: data acquisition

Pre-approved. Each strategy's acquisition runs immediately before that
strategy's design step (see "Execution").

| Priority | Item | For | Notes |
| --- | --- | --- | --- |
| 1 | 2009 OES-based AEWR schedules by area, occupation and skill | A2 | Archived FLC Data Center (Wayback); verify against 2010 rule Table 1 and `aewr_2009_nonstatutory_wages` |
| 1 | FY2006–07 H-2A disclosures | A2 pre-trends | OFLC performance-data archives |
| 1 | 2023–24 OEWS occupation AEWRs | A3 | DOL/OFLC AEWR pages and the Federal Register |
| 2 | HUD Fair Market Rents, county, FY2008–2026 | C1 | Public bulk files |
| 2 | EIA diesel retail prices (PADD regions, weekly or monthly) | C2 | Public API or bulk CSV |
| 2 | SWA prevailing-wage findings | D2 | ETA-232 / AOWL / FLAG; coverage uncertain |
| 3 | Consular processing data | C3 | Only if an origin linkage exists |

**Each item:**
- raw files under `data/raw/<source>/`;
- a sourced table, with a notes file listing citations and confidence;
- a coverage check against what the strategy needs.

If an item is unobtainable after a bounded search (about one agent-hour),
log it and kill or downgrade the strategy that needs it.

## Phase 3: per-strategy design

Each surviving strategy gets one implementer:

1. **Support statistics:** dose SD after FE, units, effective clusters,
   leverage.
2. **Confirmation split** if the strategy needs tuning (integrity rule 2).
3. **Registry:**
   - estimand, sample, dose, fixed effects, inference (size-valid primary),
     outcomes;
   - pre-period diagnostics, placebos, sensitivities;
   - prior exposure;
   - the unit-conversion assumptions.

   Freeze it and record the SHA-256 in the ledger.
4. **Design-stage power** with the shared harness.
   - **Kill** if the size-valid MDE exceeds 1.0 and the strategy cannot be
     pooled.
   - **Flag** if it exceeds 0.5.

   Strategies that pass proceed straight to Phase 4. There is no user
   checkpoint.

## Phase 4: falsification

Run the declared pre-period event studies, placebo dates, placebo groups
(D2 non-binding jobs, unchanged occupations in A3, non-range jobs in A4) and
dose-correlation checks against pre-period covariates.

A strategy that fails a declared falsification test is reported and
excluded from pooling. It is not repaired after the fact unless the repair
is labeled post-outcome.

## Phase 5: estimation

Run only what the frozen registries declare, on the confirmation split where
one exists. Write estimates, conversion to the primary estimand and
inference to the ledger.

## Phase 6: adversarial review

A fresh agent that did not build the strategy reviews each estimated
strategy. It gets:
- the registry, code and outputs;
- the threat list;
- **no** summary from the implementer.

The reviewer:
- tries to break the result (alternative mechanisms, data errors, unit
  conversions, inference);
- may run new checks, which are labeled post-outcome;
- writes `review.md` with a verdict:

| Verdict | Meaning |
| --- | --- |
| stands | No material problem found |
| stands with caveats | Usable, with the stated caveats |
| fails | Not usable |

Implementer and reviewer get one exchange. Unresolved disagreements go to
the user.

## Phase 7: synthesis

- **Per-strategy table** in the common unit, with each strategy's
  assumption.
- **Pooled estimate** under the Phase 0 rule, with a heterogeneity test and
  Romano–Wolf adjusted p-values.
- **Welfare-ready range** for `markdowns/model_sketch.md`.
- **Ledger summary:** strategies proposed, killed at each phase (with
  reasons), estimated, surviving review.
- **Report:** `markdowns/identification_search_report.md`.

## Execution: one sequential workflow

**How it runs:** the main Claude session drives the steps itself. It starts
**one subagent at a time**, in the order below, and starts the next only after
the previous one has finished and its output has been checked. No parallel
workflow is used. Every agent is a Claude agent.

1. **Phase 0 lead.** Shared outcomes, cost unit, power harness, split
   helper, declarations.
2. **Triage, credibility-first.**
3. **Triage, power-first.** Must not read step 2's output.
4. **Triage merge.** Kills a seeded strategy only if both triage agents
   killed it, adds at most two new strategies, and logs the rest as not
   pursued.
5. **Strategies, one at a time, each run to completion before the next
   starts.** Each strategy goes through:
   - acquisition, if needed;
   - design: support statistics, frozen registry, design-stage power,
     proceed or kill;
   - estimation, only if the design proceeds;
   - blind review, only if estimated.

   Order: strategies on local data first (A1 with D1, A4, B, F1), then
   those needing downloads (A2, A3, C1, C2, D2), then up to two new
   strategies from triage.
6. **A6 stacking**, if at least two A-family registries were frozen.
   - The agent reads only registries and code, never per-event estimates.
   - It freezes its own registry before estimating.
   - A blind review follows.
7. **Synthesis.** Ledger, pooled estimate, multiplicity adjustment, and
   `markdowns/identification_search_report.md`.
8. **Report check.** Every number traced to its file; factual errors
   corrected and listed.

**Running estimate:** about 30–45 agent runs. Killed strategies skip
estimation and review.

**Partial results:** each strategy's `ledger_row.json`, `estimates.csv` and
`review.md` appear as that strategy finishes, so results are readable before
the run ends.

**Stopping:** the user can stop the run at any point. The main session
resumes from the last completed step; each strategy's files record how far
it got.

## What success and failure look like

- **Success:**
  - at least two strategies with different assumptions survive
    falsification and review;
  - each has MDE ≤ 1.0, and the pooled MDE is ≤ 0.3–0.5;
  - the result separates a conventional elasticity from zero, or shows a
    precise zero.
- **Partial success:** one A-family strategy with MDE ≤ 0.5 and clean
  falsification. That becomes the headline, with the others as support.
- **Failure:** no strategy reaches MDE ≤ 1.0 with clean falsification. The
  paper then reports the Track 1 lower bound with structural (F-family)
  calibration, labeled model-based. The ledger documents that the search was
  broad and pre-declared. That record is itself a defense against
  specification-search criticism.
