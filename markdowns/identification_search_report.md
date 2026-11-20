# Identification search: report

- **Plan executed:** `markdowns/identification_search_agent_plan.md` (Phase 7 synthesis).
- **Date:** 2026-11-06.
- **Author:** synthesis agent (Claude). Every agent in this search was a Claude agent.
- **Scope:** reads files only. No new outcome contrast was estimated for this report.

`$S` means the scratch root
`/tmp/claude-1000/-home-ken-Dropbox-projects-H-2A-Paper/1be305e6-0442-48c5-93f4-7023961621fb/scratchpad/identification_search`.
`$T1` means the earlier exploration's scratch root (same path, ending
`.../scratchpad/identification`). Repository paths are relative to the root.

**Files produced by the synthesis**

- **Ledger:** `explorations/identification_search/ledger.csv`, 36 rows, one per strategy, catalogue entry and triage proposal.
- **Scripts:** `explorations/identification_search/synthesis/01_pooling_multiplicity.R` and `02_ledger.R`.
- **Outputs:** `$S/synthesis/`, which holds `hash_check.csv`, `pooling_decision.csv`, `pooled_estimate.csv`, `multiplicity.csv`, `track1_reference.csv` and `ledger_summary.csv`.
- **Hash checks:** `declarations.json` and the five frozen registries (A1, F1, A2, A3, N1) all match their recorded SHA-256 (`$S/synthesis/hash_check.csv`).

## 1. Verdict

**The search failed by the plan's own definition.**

- **No strategy reached a size-valid MDE of 1.0 or less** in the primary estimand, the elasticity of certified hours with respect to H-2A cost per hour (ε_cost). None reached 0.5.
- **No pool exists.** No strategy meets the declared inclusion rule, so no pooled estimate was computed.
- **What we know about the AEWR effect on H-2A hours is unchanged.** It is still the earlier Track 1 result, which this search did not overturn or improve. One season after the 2025 cut, there is no detectable response of hours, and the data rule out only very elastic demand: about −1 or more negative, measured against the statutory rate.

### The plan's three outcomes, applied

| Outcome (plan, "What success and failure look like") | Met? | Why |
| --- | --- | --- |
| **Success:** two or more strategies with different assumptions survive falsification and review, each with MDE ≤ 1.0, pooled MDE ≤ 0.3–0.5 | **No** | No strategy survived review. The two estimated strategies, A1 and F1, both received "fails" (`A1/review.md`, `F1/review.md`). |
| **Partial success:** one A-family strategy with MDE ≤ 0.5 and clean falsification | **No** | No A-family strategy had a design MDE ≤ 0.5. A1's 0.82 was the lowest (`A1/ledger_row.json`). |
| **Failure:** no strategy reaches MDE ≤ 1.0 with clean falsification | **Yes** | See the note on A1 below. |

**A1 comes closest to the letter of the success test but fails its substance.**

- Its design MDE was 0.82, and its gating falsification tests passed.
- But the 0.82 assumed a pass-through π of 1 from the dose to wages. The estimated first stage is π̂ = 0.033 (SE 0.101, F = 0.10) (`$S/A1/a1_conversion.csv`).
- So the realized ε_cost confidence set is the whole real line, and the realized MDE in ε_cost is 173 on the jackknife.
- The reported balance check F6 also failed: the dose correlates −0.59 with the FY2025 field-SOC share within states (`A1/ledger_row.json`).
- The review concludes that A1 does not identify the estimand (`A1/review.md` §1–4).

**The plan's failure branch:** report the Track 1 lower bound, with structural (F-family) calibration labeled model-based.

- F1, the only F-family strategy run, failed review.
- Only its descriptive pieces survive (section 3.2).
- So the calibration rests on Track 1 plus those descriptives.

## 2. Why each strategy failed

### 2.1 The pattern

The same four constraints recur.

**1. State-level shocks force the shared-shock gate.**

- The declared gate rule (`shared/declarations.json`, `power.gate_dgp_rule`): if the share of residual variance at state × FY cells is more than 1.5 times what pure unit noise produces, the gate uses the worst shared-shock error process.
- Every FY2026 design, and A3, exceeded that ratio:
  - cross-state county designs: 2.1 (3.47% vs 1.62%; `$S/triage/merged.md`);
  - A1 within state: 1.64;
  - A3: 2.17;
  - N1: 12.7.
- Under the shared-shock process, cross-state FY2026 contrasts gate at 2.3–7.9 (triage merge).
- Only within-state contrasts (state × FY fixed effects) got near 1.0. A2 alone gated on unit noise (ratio 1.35).

**2. Little dose variation is left within states.**

- Moving within state solves the shock problem, but there is little dose left once truncation by other floors, controls and timing are applied:
  - **A2:** within-state SD of the OES-area floor is $0.14 after fixed effects, over 13.3 effective states. Minimum-wage truncation affects 64% of pre-period hours in 2008–09 (`A2/ledger_row.json`, `$S/A2/a2_support.csv`).
  - **A3:** the occupation-group controls needed against the pre-rule non-field trend remove 79% of the dose variance. That leaves an SD of 0.0071 log points across 12.4 effective states (`A3/ledger_row.json`).
- The triage kills had the same root cause:
  - **C1 housing:** SD $0.043/hour after FE.
  - **C2 transport:** SD $0.005/hour.
  - **A4 range herders:** cross-state log SD 0.044.
  - **B family:** region-year only, 17 clusters (`$S/triage/merged.md` §1).

**3. Pass-through from the rule to the binding floor is weak or untestable.**

- **A1:** the within-state binding-share dose moved no wages (π̂ = 0.033).
- **N1:** the skill-tier contrast has a precise reduced form, with an ITT MDE of 0.60 at π = 1. But DOL's own projection implies π ≈ 0.138, which gives an MDE of 4.35. The break-even π is 0.60 (`$S/N1/n1_power_gate_R1.csv`).
- **A2:** designed at π = 0.4, giving an MDE of 6.95 (2.78 even at π = 1).
- **A3:** assumed π = 0.8 because it could not be measured before estimation.
- **The one strong first stage in the record is the cross-state statutory cut.** Track 1 gives 0.66, and A1's review gives 0.61 on A1's frame (`$S/A1/review/r01_r3_first_stages.csv`, row R3f, post-outcome). Cross-state contrasts are exactly the ones the shared-shock gate rules out.

**4. Floors are measured badly.**

- The shared at-AEWR flag omits the 2023-rule occupation AEWRs. This made A1's binding share wrong for non-field jobs: they are 3.5% of hours but 50.4% of the "not at AEWR" hours (`A1/review.md` §2).
- Multi-state itinerary floors and stale prior-year AEWRs make floor-bound offers look like market wages (`F1/review.md` §3–4). See section 3.4.

### 2.2 One row per strategy that reached design, plus the triage kills

Full detail, including every merged and not-pursued proposal, is in `explorations/identification_search/ledger.csv`.

MDEs are size-valid elasticity MDEs in ε_cost under the declared gate error process (DGP), unless stated otherwise.

| Strategy | Family | Phase reached | Decisive number | Reason |
| --- | --- | --- | --- | --- |
| A1: 2025 rule, within-state binding share × effective floor cut | A (with D1) | Review: **fails** | π̂ = 0.033 (F = 0.10). ε_cost set = whole line. Design MDE 0.82 assumed π = 1 | Binding share mismeasured. "Revealed binding" premise false: non-binding employers cut offers as much ($2.39 vs $2.32). The one significant secondary (SD5) is an artifact, with a placebo of the same size (`A1/review.md`) |
| F1: market-wage recovery from above-floor offers | F (model-based) | Review: **fails** | P1 θ = 0.573 [0.052, 1.094] becomes 0.61 [−0.20, 1.42] (indeterminate) after the multi-state correction | Frame conditions on the FY2026 outcome. Itinerary floors are missed. P3 rejection depends on drift. P4 puts 50.5% of latent mass below the legal minimum (`F1/review.md`) |
| A2: 2009 OES switch and 2010 return | A | Design: **killed** | 2.78 at π = 1; 6.95 at π = 0.4 | Within-state dose SD $0.14 after FE, 13.3 effective states. No method passes the size gate under shared DGPs (`A2/ledger_row.json`) |
| A3: 2023 occupation rule (non-field SOCs) | A | Design: **killed** | Gate MDE undefined (no size-valid method under the worst shared DGP). 5.52 (pair DGP) and 2.05 (unit noise) at π = 1 | Occupation controls leave 0.0071 log-point dose SD. External-validity flag (non-field jobs, 2.19% of pre-rule hours) (`A3/ledger_row.json`) |
| N1: predetermined skill tier, 2025 rule | new (merge of cred N1 and power P3) | Design: **killed** | 0.60 at π = 1; 4.35 at π = 0.138; break-even π = 0.60 | Conversion, not the reduced form. The rule's own projection puts few workers at skill II (`N1/ledger_row.json`) |
| A4: range herder phase-in | A | Triage: killed | Cross-state log dose SD 0.044; one national series | No size-valid inference; different occupation |
| B1–B3: FLS-based instruments | B | Triage: killed | Region-year only (17 clusters); MDE 1.5–25 | Plan's kill rule for B |
| C1: housing (FMR) | C | Triage: killed | Dose SD $0.043/hour; rough MDE 3 to >10 | Cents per hour; rents track local labor demand |
| C2: transport (distance × diesel) | C | Triage: killed | Dose SD $0.005/hour; rough MDE >5 | Diesel shifts all farm costs |
| A5, C3, C4, G | A, C, G | Triage: not pursued | A5 rough MDE >10; C3 has no origin field; C4 national only; G border MDEs 3.7–7 | Both passes killed (`$S/triage/merged.md` §4) |
| N4/P5, N5, P6: new proposals | new | Triage: killed | N4/P5 rough MDE > 2 (cred) to 7.6 / > 10 (power); N5 dose $0.035/hour; P6 one unit, ≤ 13 placebo dates | Killed by their own proposers (`$S/triage/merged.md` §4) |
| D1, E1–E5, cred N1–N3 and N2b, power P1–P4 | D, E, new | Triage: merged, layer or not pursued | N2b standalone gate ITT 3.94 (about 4.8 in cost units) | D1, N2/P2 into A1; N1/P3, N3 and P4 into N1; P1 into A3; N2b an A1 secondary; E1–E5 are outcome layers inside each registry (`ledger.csv`) |
| D2: other-floor placebo groups | D | Triage, then not run | — | Falsification module with no surviving host (lead decision, `D2/ledger_row.json`) |
| A6: stacked A events | A | Not run | — | Lead decision (see section 6.2) |
| F2: adoption model | F | Deferred, not run | — | Its holdout events (A2, A3) and its input (F1) did not survive |

Triage sources: `$S/triage/merged.md`, `credibility.md` and `power.md`.

## 3. What survives and is usable

### 3.1 Track 1 (pre-search; unchanged by this search)

Track 1 is the 2025 statewide rule, FY2025 vs FY2026 Q1–Q3. It compares 2,921 counties in 46 states, with county and former-region × FY fixed effects. The source is `markdowns/alternative_identification_exploration.md` §3, numbers from `$T1/track1_post_estimates.csv`.

**Setting**

- **One season**, mostly a cut: the mean dose was −$3.96.
- **First stage:** offered wages fell $0.66 per statutory dollar (jackknife 95% CI [0.47, 0.85]).
- **Placebo** (FY2024 vs FY2025): clean.

**Quantity response.** None is detectable. Elasticities are measured against the statutory U.S. skill-I rate, with jackknife 95% CIs:

| Outcome | Elasticity [95% CI] | Realized MDE |
| --- | --- | --- |
| Certified positions per 2011 farm worker (equal-county) | 0.40 [−1.06, 1.86] | 2.03 |
| Certified hours per 2011 farm worker | 0.21 [−0.85, 1.27] | 1.48 |
| PPML certified positions | −0.12 [−0.77, 0.53] | 0.90 |
| **PPML certified hours** (closest to this search's primary) | **−0.20 [−1.00, 0.61]** | **1.12** |

**What it rules out:** elasticities with respect to the statutory rate below about −1. It does not separate conventional values (around −0.3) from zero.

**Track 1 in this search's unit.** This is a synthesis conversion, done after the fact and not declared by any registry. Results are in `$S/synthesis/track1_reference.csv`.

- **Method:** divide by the first stage (0.66) and scale by the cost-to-wage ratio (1.2401) at the mean offered wage ($16.60). Uncertainty in π is ignored.
- **Result for PPML hours:** ε_cost = −0.40 [−2.03, 1.23], with an MDE of about 2.3.
- **Implication:** in the primary estimand, Track 1's lower bound is about −2, not −1. The "about −1" bound is a statement about the response to the statutory AEWR, which is the policy variable. It is not a statement about cost per hour.

### 3.2 F1 descriptives the review lets through

The source is `F1/review.md`, "Use in the synthesis". The scope is FY2026 field, hourly, non-piece orders received from 2025-10-02 to 2026-06-30, in 48 states.

**Where hours sit relative to legal floors**

- **At a legal floor:** b26(F_app) = 0.851 [0.802, 0.895] (`$S/F1/estimates.csv`, `P2_b26[F_app]`).
  - Another ~3% sit at a floor above F_app, mostly skill-I orders paying the U.S. skill-II rate.
  - So about 0.88 of hours are at some floor.
- **At the old FY2025 AEWR:** 4.3% of hours.
- **Strictly above every floor:** 6.0–7.7%, depending on the multi-state correction.

**Pass-through above the floor is partial.**

- θ_T = 0.39 [0.24, 0.56], or 0.35 [0.15, 0.57] with the multi-state correction. These are secondary, assumption-labeled numbers.
- The cross-state θ ranges from 0.13 to 0.61, depending on frame, dose and classification.
- Pure "fixed market wage" (θ = 0) is contradicted by units that cut down to the new floor.
- Pure "fixed premium" (θ = 1) is rejected in most variants.
- The reviewer's defensible scenario range for dw/da above the floor is about [0.15, 0.6].

**Exposure range for the welfare model**

- x(F_app) ≈ 0.90–0.94 of hours, with polar cases 0.85 and 1.0.

**Not usable**

- Any recovered market or alternative-wage distribution, including P4 below the floor.
- The registry's mechanical welfare feed "dw/da = 1 for all hours".
- Any elasticity.

### 3.3 Data assets acquired (usable by later work)

| Asset | Where | Built by |
| --- | --- | --- |
| 2009 OES-based AEWR schedules (OWL 2008–2010, area × occupation × skill), OES area–county crosswalk, verification | `data/raw/aewr_2009_oes/`; `explorations/identification_search/sources/aewr_2009_oes_schedule.csv`, `oes_area_county_crosswalk_2009.csv`, `aewr_2009_oes_verification.csv`, `aewr_2009_oes_notes.md` | `A2/00`–`01` |
| FY2006–07 H-2A disclosures | `data/raw/h2a_disclosures_fy2006_2007/`; `sources/h2a_disclosures_fy2006_2007.csv` | `A2/02` |
| 2023–2025 occupation (OEWS) AEWRs by vintage, field-group rates, regime periods, offer-match checks | `data/raw/aewr_2023_oews_occupation/`; `sources/aewr_2023_2025_*.csv`, `aewr_2023_2025_occupation_notes.md` | `A3/00`–`01` |
| Kansas v. DOL injunction scope (17-state proxy, 14-day lag) and Teche Vermilion injunction and vacatur scope and dates | `sources/aewr_2023_2025_occupation_notes.md` §2–3; court PDFs in `data/raw/aewr_2023_oews_occupation/court/` | A3 acquisition |
| FY2025–26 state agricultural minimum wages for H-2A (fixes FL 2025-09-30, NY upstate/downstate, OR tiers), sourced per row | `explorations/identification_search/A1/sources/state_ag_minwage_fy2025_2026.csv`, `minwage_substate_counties.csv` | `A1/00` |
| Experience-requirement extract (`WORK_EXPERIENCE_MONTHS`, FY2020–FY2026 Q3; valid on 99.0% of FY2020 cases and about 100% after) | **Scratch only:** `$S/N1/n1_experience_extract.parquet`, coverage in `$S/N1/n1_experience_coverage.csv`. Rebuild with `N1/01_experience_extract.R` | `N1/01` |

### 3.4 Measurement warnings for the shared tools

These come from reviewer and designer findings. They apply to any later use of `explorations/identification_search/shared/` outputs or similar flags.

1. **The at-AEWR flag (`share_at_aewr`, `at_aewr_hours`) omits occupation floors.**
   - It compares offers only with the field/livestock reference AEWR. From 2023-03-30 to 2025-08-28, non-field SOCs had their own OEWS AEWRs.
   - Non-field jobs are 3.5% of FY2025 hours but 50.4% of "not at AEWR" hours, and 86% of their hours sit at a state × SOC modal wage.
   - Source: `A1/review.md` §2. The acquired A3 rates now make a fix possible.
2. **The flag also misclassifies below-AEWR hours.**
   - 55% of the field hours flagged "not at AEWR" are more than $0.05 *below* the FY2025 AEWR. Louisiana alone holds 29% of these. The cause is unresolved (`A1/review.md` §2).
3. **Multi-state itinerary floors.**
   - An itinerary order pays one wage, typically the highest floor among its states. Per-state classification then calls rows in lower-floor states "above every floor".
   - This affects 42.8% of FY2025 and 21.5% of FY2026 "above" hours (`F1/review.md` §3).
4. **Stale AEWRs.**
   - Prior-year AEWR values persist as offers. Example: $18.50 in MN and MI, the 2024-01-01 AEWR.
   - Among FY2026 offers, 4.3% of hours sit at the old FY2025 AEWR (`F1/review.md` §4, §3.2 above).
5. **Employer-ID break at FY2020.**
   - From FY2020, Addendum B names worksite businesses, so active employers jump from 8,459 (FY2019) to 26,284 (FY2020).
   - County geography also widens at FY2015 and FY2020 (`shared/declarations.json`, `outcomes.known_limitations`).
6. **Weekly hours are missing for every FY2008–FY2010 case.**
   - Hours there are positions × weeks × a pooled mean, so the weekly-hours margin carries no information (same source). A2 used a constant 40 hours.
7. **Disclosed offers are U.S.-worker rates.**
   - From 2025-10-02 the H-2A cash wage, net of the housing adjustment, is not observed except where a minimum wage exceeds the U.S. rate (`$S/triage/merged.md`, A1 note 6).

### 3.5 Legal-history facts found

- **Teche Vermilion SCGA v. Su** (W.D. La., No. 6:23-cv-831).
  - The preliminary injunction is dated 2024-09-18 and was amended 2024-12-05. It covers Louisiana sugar cane employers in named associations.
  - The final judgment of 2025-08-25 (filed 2025-08-26) made the injunction permanent and **vacated the 2023 rule**.
  - OFLC announced on 2025-08-28 that AEWRs follow the 2010 method. FLAG listed one FLS AEWR per state for all non-range occupations from 2025-08-29.
  - So the non-field "2025 cut" happened on 2025-08-29, not 2025-10-02 (`sources/aewr_2023_2025_occupation_notes.md` §3; court PDFs and the OFLC announcement copy in `data/raw/aewr_2023_oews_occupation/`).
- **Kansas v. DOL** (S.D. Ga.): an entity- and state-limited injunction against the 2024 Farmworker Protection Rule, with a 14-day lag on AEWR updates in 17 states (same notes, §2).
- **A reported court ruling against the 2025 rule, with an OFLC notice dated 2026-09-02. The notice's existence is documented; the ruling's content is unverified.**
  - One scratch file documents the notice: the FLAG "General Announcements" page captured 2026-11-05 during A3 acquisition (`$S/A3_dl/flag_ann.html`; not copied to `data/raw`). Its September 2, 2026 entry reads: "OFLC Announces Updates to Implementation of the H-2A Adverse Effect Wage Rate Methodology for Non-Range Occupations Interim Final Rule; Compliance with District Court Order. This notice is provided to comply with the August 26, 2026 Order of the United States District Court for the Eastern District of California entered in United Farm Workers, et al. v. DOL, et al., No. 25-cv-01614-KES-EGC ("Order"), requiring the U.S. Department of Labor (the "Department") to provide..." The excerpt is truncated there.
  - No file holds the order, the docket or the notice's full text. So what the order requires, and whether it is a ruling against the 2025 rule, is unverified.
  - Other traces:
    - N1's registry refers to "the later court-ordered methodology" as outside its data window (`N1/registry.json`, assumption A-price).
    - The FLAG AEWR page captured 2026-11-05 (`data/raw/aewr_2023_oews_occupation/flag/flag_aewr_page_20261105.html`) is internally mixed. It is headed "Last Updated: August 3rd, 2026". It describes non-range AEWRs as set by the USDA FLS method and lists FLS rates with December 2024 effective dates under "Current AEWRs". It also links "statewide 2026-2027 IFR AEWRs" and archives the 2025–26 IFR rates as "Effective October 2, 2025 through August 2, 2026".
  - This conflicts with the premise recorded for Track 1, which you confirmed on 2026-11-02, that legal challenges to the 2025 rule ultimately failed (`markdowns/alternative_identification_exploration.md` §3).
  - **Why it matters:**
    - FY2026 Q1–Q3 is not affected: the archived IFR rates are labeled effective through 2026-08-02.
    - It would matter for any use of FY2026 Q4 or FY2027 data, and possibly as a new event (section 5).
  - **It needs verification against the court docket and the OFLC notice's full text before anyone relies on it.**

## 4. Welfare-ready range for `markdowns/model_sketch.md`

The model's central behavioral object is the response of certified contract hours to the AEWR (`model_sketch.md` §3–4). Its welfare bridge is dΠ/da = −X_m(a), where X_m is AEWR-exposed hours (§3.2 there).

**What the evidence supports**

| Input | Range supported | Basis | Status |
| --- | --- | --- | --- |
| Hours response per statutory AEWR dollar (PPML semi-elasticity ψ = −η) | η = −0.013 per $; 95% CI [−0.065, +0.040] | Track 1 PPML hours (`$S/synthesis/track1_reference.csv`) | Credible design, one season |
| Same, as an elasticity with respect to the AEWR | −0.20 [−1.00, 0.61]; **lower bound about −1.0** | Same | Credible lower bound. No credible point estimate |
| Same, in ε_cost (response to cost per hour) | −0.40 [−2.03, 1.23] | Post-hoc conversion: π = 0.66, ratio 1.24 | Lower bound about −2 |
| Share of hours exposed to marginal floor changes, x | 0.90–0.94 central; 0.85 and 1.0 polar | F1 review descriptives | Descriptive, FY2026 field hourly non-piece |
| Pass-through above the floor, dw/da | about 0.15–0.6; 1.0 as the upper polar case | F1 review | Assumption-labeled |

**Recommended use**

- Report welfare over a scenario grid for the hours elasticity with respect to the AEWR: 0, −0.3, −0.75 and −1.0. Treat −1.0 as the most elastic case the data support.
- Do not present any value inside [−1, 0] as estimated.
- **The first-order envelope term barely depends on the elasticity.** It is fixed by observed exposed hours, X0 ≈ 0.85–1.0 of certified hours. The elasticity enters only through the change in X along the path. So for small AEWR changes the welfare range is much tighter than the elasticity range. This follows from the model's integral; it is not a new estimate.

**Assumptions behind the range**

- **Horizon:** one season. Contracts and crop choices may adjust over longer horizons.
- **Direction:** the variation is mostly a cut (mean −$3.96). The response to an increase may differ.
- **Timing:** the rule took effect on publication, 2025-10-02. `model_sketch.md` asks for an AEWR announced before filing, so this is a shorter-run experiment than the model's primary counterfactual.
- **Unit:** the elasticity is with respect to the statutory rate. Pass-through to offered wages was 0.66.
- **Confounding:** no unmeasured FY2026 shock correlated with the within-region dose. Immigration enforcement is the named threat.
- **Legal status:** the rule applied throughout FY2026 Q1–Q3 (section 3.5).

## 5. What would change the picture

None of these guarantees an MDE of 0.5 or less. Each removes a binding constraint named in section 2.

1. **FY2026 Q4 and FY2027 disclosures, once released.**
   - A second season of the 2025 statewide rule, and the 2026-08-03 schedule as a new cross-state event, would add independent periods of dose variation.
   - The binding constraint is shared state × FY shocks, so new seasons help only if their dose pattern differs across states.
   - Track 1's PPML-hours MDE (1.12 in AEWR units) would need about five independent seasons of similar variation to reach 0.5. That is simple √n arithmetic and assumes independent shocks.
2. **The reported court ruling, if verified.**
   - A court-ordered return to the FLS method after the 2025 rule would be an opposite-signed statewide event with a known cross-state pattern.
   - A symmetric reversal is the cleanest available defense against state-specific trends.
   - It would need its own registry before any post-ruling data are read.
3. **Outcomes with less noise at the unit of identification.**
   - Receipt-dated requested positions did not help: A1's lead-secondary design MDE was 1.09 (`$S/A1/a1_power_summary.csv`).
   - Job-order-level timing around a rule date (E1) or employer × worksite cells (E3) would help only if shared shocks are smaller at that level. A1's employer-level secondary (SD5) shows that composition artifacts can dominate there (`A1/review.md` §5).
4. **Fixing the floor measurement** (section 3.4).
   - Rebuild the binding flag with occupation AEWRs, itinerary floors and stale-AEWR detection.
   - This is a precondition for any revealed-binding (D1) design. It would not by itself rescue A1, because A1's review found that above-floor employers also cut their offers.
5. **A measured skill-tier assignment response (N1).**
   - N1's reduced form is precise (ITT MDE 0.60 at π = 1). What it lacks is pass-through: break-even π is 0.60.
   - If FY2026 data showed tier pass-through near that level, N1 could be revisited. The rule's own projection (0.14) and the triage take-up evidence (π about 0.1–0.27; `$S/triage/merged.md` §3) point the other way.

## 6. Integrity notes

### 6.1 Prior exposure: strategies that were not outcome-blind

Each item lists what had already been seen; the sources are each strategy's registry and `ledger_row.json`, `prior_exposure` field.

- **A1:**
  - Track 1 had already seen FY2026 Q1–Q3 certified positions, hours and equal-county ratios by county, plus the cross-state first stages.
  - The credibility triage had seen FY2026 certified positions and offered wages by FY2025 experience class.
  - The within-state binding contrast and receipt-dated (rq13d) FY2026 hours were new.
- **F1:** **P1 was not blind.**
  - A1's review had already shown that FY2025 field incumbents with low binding shares cut offers about as much as binding ones. That points toward θ near 1.
  - F1's designer recorded this in the registry.
- **N1:** the designer had read the A1 and F1 estimates and reviews, including the FY2026 offer distribution by experience, and the credibility triage's take-up tabulations. N1 was killed before any outcome contrast.
- **A3:**
  - The power triage printed the national non-field share of certified positions through FY2026 Q1–Q3, which is the treated group's post-period.
  - The A3 designer printed national rq13d hours FY2020–FY2026 while checking coverage.
- **A2:** the earlier 2009–10 exploration had estimated a county response that looked like a trend. The power triage calibrated on post-event FY2011–14 variance.
- **Triage:** the credibility pass viewed FY2026 Q1–Q3 certified positions by FY2025 experience class (used as weights).

### 6.2 Deviations from the plan

- **A6 was not run, although the plan's literal condition was met.**
  - The plan says to run A6 "if at least two A-family registries were frozen". A1, A2 and A3 all froze registries.
  - The lead instead required registries "that proceeded past design" (`A6/ledger_row.json`).
  - No stacked design-stage MDE was computed, so whether A6 could have reached 1.0 is unknown.
  - My judgment, not a computed result: the component support statistics make it unlikely. A2 is 2.78 even at π = 1, A3 has no size-valid gate MDE, and A1's within-state dose has no first stage.
- **D2 was not run** (lead decision: no surviving host). The SWA prevailing-wage search was never attempted, and no D2 registry exists.
- **F2 was not run** (deferred at triage; its inputs did not survive).
- **A1 changed its primary contrast** from the catalogued cross-state contrast to the within-state D1 contrast. This was declared in the registry (`change_from_catalogue`) and chosen from support statistics only. A1 also did not run one declared alternate inference (SD2 randomization inference); `A1/review.md` §7 treats this as immaterial.
- **F1 implementation**, per `F1/review.md` §1:
  - The registry is internally inconsistent on the P1 frame (285 vs 352 units).
  - `03_est_helpers.R` was edited after `04` ran; the reviewer reproduced every P1 number and the P3 and P4 point estimates exactly.
  - `09` was rerun.
  - The implementer reported no multiplicity-adjusted p-values.
- **No per-criterion rubric scores exist for N1.** The merge recorded only totals.
- **The A3 ledger row was written by hand** from the power outputs.

### 6.3 Pooling and multiplicity

**Pooling.** The declared rule (`shared/declarations.json`, `pooling_rule.inclusion`) was applied criterion by criterion (`$S/synthesis/pooling_decision.csv`). No strategy is eligible, so no pooled estimate was computed, and none is reported (`$S/synthesis/pooled_estimate.csv`).

| Strategy | Criteria failed |
| --- | --- |
| A1 | Review verdict; conversion to ε_cost (whole-line set) |
| F1 | No design MDE in ε_cost; review verdict; no ε_cost conversion; F-family is excluded from the headline anyway |
| A2, A3, N1 | Design-stage MDE above 1.0 |
| D2, A6 | No registry |

- **The sensitivity pool is empty.** It would add the external-validity-flagged strategies, but A2, A3 and A4 were killed.
- **A1's `ledger_row.json` still says `pooling_eligible: true`.** That was the design-stage value; the synthesis ledger records false.

**Multiplicity** (`$S/synthesis/multiplicity.csv`)

- **Romano–Wolf is infeasible here.** F1's P3 has only a percentile-bootstrap CI, so studentizing it in each joint draw would need a nested bootstrap. P1's size-valid inference is the jackknife. A1's state bootstrap was never size-tested.
- So the declared fallback, Holm on the size-valid p-values, was used.

| Hypothesis | Size-valid p | Holm, declared ε_cost family | Holm, extended family |
| --- | --- | --- | --- |
| A1 primary: ε_cost = 0 (state CRV1; AR/jackknife p 0.719) | 0.705 | 0.705 | 0.705 |
| F1 P1: θ = 0 | 0.032 | — | 0.097 |
| F1 P1: θ = 1 | 0.105 | — | 0.210 |
| F1 P3: ΔS = 0 (normal approximation from the bootstrap SE) | 0.0033 | — | 0.013 |

- **The declared family** (H0: ε_cost = 0 for every strategy that reached estimation) contains A1 alone. F1 has no ε_cost hypothesis, and A1 declared no co-primary outcome.
- **The extended family** adds F1's three declared gating primaries, as requested for this synthesis.
- **After adjustment,** only F1's P3 rejects. The F1 review shows that rejection is not robust: it disappears with zero drift or with multi-state floors censored (`F1/review.md` §4).

### 6.4 Reviews and post-outcome analyses

- **All reviewers were Claude agents.** So were all implementers, both triage passes, the merge and this synthesis. Independence came only from fresh agents reading files. No Codex cross-review was run; the user may add one on the same files.
- **Post-outcome analyses**, all labeled as such:
  - `A1/07_post_outcome_first_stage_descriptive.R`;
  - A1 review scripts `A1/review/r01`–`r03`, including the cross-state π = 0.61 and the SD5 placebo;
  - F1 PO1 (the 352-unit frame) and PO2;
  - F1 review scripts `F1/review/r01`–`r05`;
  - the Track 1 ε_cost conversion in this report (section 3.1).
- **Review findings that change the credibility of shared tools:**
  - The at-AEWR flag omits occupation floors and misclassifies below-AEWR field hours (A1 review).
  - Itinerary and stale-AEWR floors (F1 review).
  - The triage's "one in three FY2026 hourly positions above the floors" came from that flag, which ignores minimum wages. Under F1's classification the figure is 7.6% of FY2026 field non-piece hours (`F1/ledger_row.json`).
  - **Phase 0's harness test** did not apply the node-share gate rule to the PPML-weighted outcome (`$S/triage/merged.md`, gate section). The triage merge corrected this, and every later designer recomputed the rule.

## Corrections by checker

Independent report check (plan Execution step 8), 2026-11-06. Each fix below was traced to the cited file.

1. **Section 3.5, the 2026 court ruling.** The report said no file in the repository or in `$S` documents the ruling or a 2026-09-02 notice. That was wrong: `$S/A3_dl/flag_ann.html` (FLAG General Announcements, captured 2026-11-05 during A3 acquisition; scratch only) lists the September 2, 2026 OFLC notice, citing an August 26, 2026 order of the E.D. Cal. in *United Farm Workers v. DOL*, No. 25-cv-01614-KES-EGC. The excerpt is truncated, and no file holds the order or the notice's full text, so the ruling's content stays unverified. The bullet now says this.
2. **Section 2.1, item 1.** A3 is not an FY2026 design (its calibration is FY2020–22). "Every FY2026 design exceeded that ratio" now reads "Every FY2026 design, and A3, exceeded that ratio".
3. **Section 2.2, F1 row.** The multi-state correction moves P1 θ from 0.573 to 0.612, so "falls to 0.61" now reads "becomes 0.61 … (indeterminate)" (`F1/review.md` §3).
4. **Section 2.2, missing ledger entries.** The table claimed to list the triage kills but omitted cred N4, cred N5, power P5 and power P6. Merged and layer entries (D1, E1–E5, cred N1–N3, cred N2b, power P1–P4) did not appear anywhere in the report. Two rows were added from `ledger.csv` and `$S/triage/merged.md` §3–4.
5. **Section 5, item 3.** A1's lead-secondary (requested positions) gate MDE is 1.09, not 1.08. The declared gate rule takes the largest size-valid MDE over the three shared DGPs. Those are 1.08, 1.09 and 1.09 (`$S/A1/a1_power_summary.csv`, run P3).
6. **Section 5, item 5.** The take-up evidence (π about 0.1–0.27) is in `$S/triage/merged.md` §3 (the N1 section; also §5 item 6), not §2.
7. **Section 6.2, F1.** The reviewer reproduced every P1 number and the P3 and P4 point estimates exactly, not "every number" (`F1/review.md` §1).
8. **Section 6.3, pooling table.** F1 also fails criterion 2 (no design MDE in ε_cost) per `$S/synthesis/pooling_decision.csv`.
9. **Section 6.3, sensitivity pool.** A2 is also external-validity flagged (`A2/ledger_row.json`, `ledger.csv`). The list of flagged strategies that were killed now includes A2.
10. **Section 6.4.** F1's 7.6% is a share of FY2026 field non-piece hours, not positions (`F1/ledger_row.json`). The unit is now stated.
11. **`ledger.csv`, A6 `kill_reason`.** It said the plan's Execution step 6 "requires at least two A-family registries that proceeded past design". The plan requires only that "at least two A-family registries were frozen". The row now attributes the "past design" reading to the lead and says the plan's literal condition was met by A1, A2 and A3. `A6/ledger_row.json` and `$S/synthesis/pooling_decision.csv` still carry the original wording; they were not changed.

**Checked without change:**
- every other number against its cited file;
- registry and declaration hashes;
- file times showing every estimate and every design-stage power output postdates its registry;
- the pooling and multiplicity procedure against `shared/declarations.json`;
- the Track 1 numbers against `markdowns/alternative_identification_exploration.md` §3;
- the statement of the A6 deviation.
