# F1 blind adversarial review (Phase 6)

Reviewer: a fresh agent that did not build F1. I judged F1 from the frozen
registry, the code in `explorations/identification_search/F1/` (00–11), the
outputs in `$S/F1/`, the shared tools, the plan and the A1 review. I ignored
the implementer's interpretive prose. Every number below is either read from
an output file or recomputed by my own scripts. My scripts are in
`explorations/identification_search/F1/review/` (r01–r05) and their outputs
are in `$S/F1/review/`. All of my checks are **post-outcome**.

`$S` = `/tmp/claude-1000/-home-ken-Dropbox-projects-H-2A-Paper/1be305e6-0442-48c5-93f4-7023961621fb/scratchpad/identification_search`

## Verdict: **fails**

F1 does not deliver what it was built to deliver: a recovered market-wage
distribution, a decision between M1 (fixed market wage) and M2 (fixed premium),
and a welfare feed. Three problems cause this:

1. **The P1 "M2_supported" reading is not robust.** It depends on a sample
   frame that conditions on the FY2026 outcome. It also depends on a floor
   classification that misses the multi-state itinerary floor. Once that floor
   is accounted for, theta = 0.61 [−0.20, 1.42], which is indeterminate.
   theta = 1 is rejected in most other variants.
2. **V1/P3 does not reject M1 robustly.** The rejection disappears when g = 0
   or when multi-state floors are censored. In addition, its null hypothesis
   (Delta_S = 0) is biased by the 27% growth in hours.
3. **P4 is incoherent as an alternative-wage distribution.** It puts 50% of
   latent mass below the legal minimum wage. Its below-floor numbers are
   artifacts of the chosen distribution family.

Some descriptive pieces survive and are listed under "Use in the synthesis".

The registry rule maps P1 + P3 to M2, and M2 implies dw/da = 1 for all hours.
That welfare feed is not supported by F1's own estimates.

## Issues

### 1. Integrity: freeze respected; readings disclosed; no Holm adjustment (minor)

- **Hashes match.** The registry SHA-256 equals the recorded `93cc085f…c2ec`.
  The hashes of `00_helpers.R`, `01_support.R` and `02_design_precision.R`
  match the registry's `code_at_freeze`. Every estimation script calls
  `f1_check_registry()`.
- **File times are in the right order.** The registry was written at 20:34:09.
  The design logs were written earlier (01 at 20:16, 02 at 20:32). The first
  estimate (04) was produced at 20:40. The P4 choice file (20:41:55) predates
  the confirmation fit (07, 20:42–20:53), and 06 refuses to rerun.
- **03_est_helpers.R was edited after 04 ran.** It was modified at 20:40:54,
  after 04 ran at 20:40:06. My reproduction with the current 03 matches every
  P1 number exactly (r01). The P3 and P4 point estimates also reproduce exactly
  (r03, r04).
- **09 was rerun at 21:35.** The code comment describes a performance refactor
  of the G3 Tobit ("identical estimator"). G3 is descriptive. I see no sign
  that the rerun was driven by results.
- **The implementer's readings where the registry was silent are mostly
  benign.** These are: INC presence on M26 orders (see issue 2); the
  INC_placebo next-FY wage; the P4 simplicity order (moot, since normal_common
  won by 25 log-likelihood units); and the V2/V3 observed shares. The P1 frame
  reading follows the literal registry text. The post-outcome alternative
  (script 10) was disclosed and labeled.
- **No adjusted p-values were reported.** This violates plan rule 4:
  `ledger_row.json` has `p_adjusted: null`. With Holm over the three gating
  primaries (P1 H0a, P1 H0b, P3), the P1 test of theta = 0 has p = 0.032,
  which becomes Holm p = **0.064** (r01_holm.csv). With adjustment, theta = 0
  is not rejected. The P1 reading would then be **indeterminate**, not M2. The
  registry rule is written on unadjusted CIs, so the mechanical reading follows
  the registry. Even so, the M2 call does not survive the plan's own
  multiplicity rule.

### 2. The P1 frame conditions on the FY2026 outcome, and the reading flips with it (major)

- **What the registry says.** Its `samples.INC` text says presence requires
  "≥ 1 new-rule M26 order". M26 is defined as the classes above, at_floor and
  at_old_aewr. Read literally, this gives **285** units. But the registry's own
  `design_stage` records **352** units "present in FY2026 new rule", and the
  size and power simulation that made the jackknife the primary inference was
  run on the 352-unit frame (`02_design_precision.R`, `semi_join` on any
  new-rule field non-piece order). The registry is internally inconsistent. The
  implementer followed the text. The design gate was passed on the other
  frame.
- **Why this is selection on the outcome.** The M26 class is a function of
  w26 relative to the FY2026 floors, and w26 is the dependent variable. The
  frame drops units whose FY2026 offers fall below the applicable floor. The
  truncation point F26 = e25 + dF moves with the dose. So truncation removes
  large cuts mainly where |dF| is small, which steepens the slope. The data
  show this:
  - The 67 dropped units have mean dw = **−3.56**, against −1.92 for the kept
    units.
  - Their dF has SD 1.94, against 1.05 for the kept units.
  - Colorado (dF = −1.56) loses 16 of its 20 units. Its state mean dw moves
    from −2.79 to −0.46 (r01_state_means.csv).
- **Reproduced estimates** (r01_p1_frames.csv; jackknife CV3J, t(G−1)):

  | Frame | theta | 95% CI | p(theta = 0) | p(theta = 1) | Rule reading |
  | --- | --- | --- | --- | --- | --- |
  | Declared 285 | 0.573 | [0.052, 1.094] | 0.032 | 0.105 | M2 |
  | Design 352 | 0.331 | [−0.054, 0.715] | 0.089 | 0.0012 | M1 (if V1 passed) |
  | 352, unweighted | 0.361 | [0.058, 0.665] | 0.021 | 0.0002 | partial |
  | 352, skill-floor dose | 0.132 | [−0.060, 0.323] | 0.17 | 1e−10 | M1 |
  | 285, skill-floor dose (implementer) | 0.211 | [−0.012, 0.434] | 0.063 | 3e−8 | M1 |
  | theta_T time contrast (implementer) | 0.388 | [0.240, 0.564] | – | – | partial |

- **Neither frame is clean.** The 67 dropped units are **98% multi-state**
  orders (see issue 3), against 50% of the kept units. Their FY2026 offers are
  "below the floor" only under a per-state classification that ignores the
  itinerary. Including them adds noise. Excluding them selects on the outcome.
- **The primary dose mismeasures the floor change for most INC hours.** 69% of
  the kept units' FY2025 hours have experience ≥ 3 months. The 3-month line
  sorts FY2026 offers sharply: 81% of exp = 3 hours are at the U.S. skill-II
  rate, against 5% for exp = 0 (r02b). For these units the skill-II rate
  governs, but the primary dose `eff_cut_u` uses the skill-I rate. With the
  skill dose, theta is 0.13–0.21, and theta = 1 is rejected decisively.

### 3. The classification misses the multi-state itinerary floor (major; affects P1, P3, P2 and V4)

- **How F1 classifies.** F1 compares each case × state row with that state's
  floors only. A multi-state itinerary order offers one wage, typically the
  highest floor among its states (custom harvest is the obvious case). F1 then
  calls the rows in lower-floor states "above every floor".
- **How much this matters (r05).** I counted "above" field non-piece rows of
  multi-state cases whose wage is within $0.05 of a floor of another itinerary
  state:
  - **FY2025: 42.8% of "above" hours** (366 orders). The chance benchmark,
    with wages shifted by ±$0.37, matches about 0.06–0.08 of the actual count.
    One example: $19.21 recurs in OK, MO, MT, TX and CO, which have different
    own AEWRs.
  - **FY2026: 21.5% of "above" hours** (599 orders). The chance benchmark
    matches about 0.10–0.12 of the actual count.
- **Effects once these rows are censored:**
  - FY2026 share above every floor: 0.077 → **0.060**.
  - INC: 285 → 151 units (30 states). The 285 vs 352 discrepancy almost
    vanishes (151 vs 153).
  - P1 theta = **0.612 [−0.197, 1.420]**, p(0) = 0.13, p(1) = 0.34. Under the
    rule this is **indeterminate**.
  - theta_T = 0.353 [0.145, 0.565]. Only the INC side is corrected; the
    FY2024 placebo base is not reclassified.
  - P3 Delta_S = −0.0040 [−0.0096, 0.0001]. **Not rejected** (and +0.0006 with
    g = 0).
- **Other classification checks passed:**
  - A hand recomputation of 40 random FY2026 orders from the source tables
    (statewide skill rates, A1 minimum-wage table, old AEWRs) reproduced
    F_app and the class in **40/40** cases (r02a).
  - The $0.05 tolerance is not fragile. Only 3.2% of "above" hours lie within
    $0.05–$0.10 of a state candidate (r02d), and S4 at $0.10 leaves every
    statistic unchanged in substance.
  - NY downstate/upstate and OR nonurban/standard are handled. Portland metro
    has no area code in `f1_area_of`, but its $16.30/$16.80 values are
    censoring candidates, so the effect is small.
  - City and county minimum wages and prevailing-wage findings remain unseen,
    as the registry acknowledges.

### 4. V1/P3: the M1 rejection is not robust, and its null is biased (major)

- **Primary result.** Delta_S = −0.0085 [−0.0151, −0.0039] rests on 332
  FY2026 orders above a25(1+g) against 608 FY2025 orders above a25 (r03a).
- **The rejection depends on the drift g** (r03b, r03d):

  | Specification | Delta_S | 95% CI |
  | --- | --- | --- |
  | Primary, g = 0.0376 | −0.0085 | [−0.0151, −0.0039] |
  | g = 0 | −0.0014 | [−0.0080, 0.0042] |

  In absolute hours, the FY2026/FY2025 ratio of hours above the cutoff is
  0.64 with g but **1.17** with g = 0 (balanced units: 0.57 vs 1.06). The whole
  result sits in the $0.66 band between a25 and a25(1+g).
- **The M1 null is wrong for this period.** M26 hours are **27%** larger than
  M25 hours (20% in balanced units). Entry and expansion at the lower floor
  dilute the upper tail even under M1. With FY2025 above-a25 hours held fixed,
  M1 predicts S26 ≈ 0.0133, not 0.0170. The dilution-adjusted null is
  Delta_S ≈ −0.0037, about half the observed gap.
- **S3 against the alternatives:**

  | Variant | Delta_S | 95% CI | Rejects? |
  | --- | --- | --- | --- |
  | S3 (all shared values censored) | +0.0002 | [−0.0025, 0.0031] | no |
  | FY2025 stale-AEWR values only | −0.0048 | [−0.0076, −0.0023] | yes |
  | S3 on non-round values only | −0.0044 | [−0.0100, −0.0002] | yes, barely |
  | Multi-state floors | −0.0040 | [−0.0096, 0.0001] | no |

- **What the shared values are (r02e).** The S3 values differ by year:
  - **FY2025:** largely real unseen floors. Examples: $18.50 in MN and MI is
    the 2024-01-01 AEWR (32% of shared hours). $19.21 recurs across eight
    states and is consistent with multi-state itinerary rates.
  - **FY2026:** largely round numbers. $15.00 appears in 20 states (23% of
    shared hours), and 50% of shared hours are at whole-dollar values. These
    look like employer choices, not floors.

  So S3 over-censors FY2026 and correctly censors much of FY2025.
- **Better-supported reading.** The FY2025 upper tail is contaminated by
  floors: stale AEWRs and itinerary rates. Once those are handled, the
  remaining evidence against M1 is weak and depends on g. V1 neither rejects
  nor supports M1 robustly.
- **Unit-level evidence is stronger.** Among the declared INC units, 105
  units (44% of FY2025 hours) were wholly above every floor in FY2025 and are
  wholly at a floor in FY2026, with mean dw = **−2.87** (r01). Among the
  multistate-corrected INC units, 63% of FY2026 hours are at a floor. Units
  that cut down to the new floor are inconsistent with a fixed market wage
  above the old AEWR.

  One caveat: for skill-II orders the new floor can exceed a25. So "at a floor"
  alone does not prove a cut; the dw values do.
- **Overall reading.** The data reject pure M1 for a large share of incumbent
  hours. They do not support pure M2 either. Partial, heterogeneous
  pass-through is the reading consistent with the most specifications (theta
  about 0.13–0.61; theta_T about 0.35–0.39).

### 5. P4: the below-floor extrapolation is unusable, even as a scenario band (major)

- **Reproduction.** Censoring is 88.6% of hours, and the chosen model is
  normal_common. My refit reproduces every output (r04).
- **The latent distribution is incoherent as an alternative wage** (r04a):

  | Quantity | Value |
  | --- | --- |
  | Latent mass below the legal minimum max(area MW, $7.25) | **0.505** |
  | Latent mass below the federal $7.25 | 0.113 |
  | Implied mean latent wage | $10.9 |
  | Implied median latent wage | $10.5 |
  | ACS 2024 state agricultural mean wage (same hours weights) | $19.2 |
  | F_app | $15.0 |

  The domestic alternative wage cannot lie below the minimum wage.
- **The family identifies the result** (r04c). A normal model fitted only to
  the observed upper tail (truncated likelihood) has scale 0.14, against 0.27
  in the censored fit. It implies P(w* ≤ F_app) = 0.59, but the observed share
  is 0.887. It also implies b(H-2A skill-I) = **0.18**, against 0.67 for P4.
  The normal family cannot fit the tail shape and the censored mass at once.
  The censored mass then fixes the location, and that drives every
  below-floor number.
- **The declared band is too narrow.** The model-uncertainty band covers only
  four nearly identical families (b(H-2A s1) in [0.635, 0.670]). It is far
  narrower than the design simulation's own misspecification bias (up to
  +0.21) and than the 0.18 vs 0.67 gap above. The band understates
  uncertainty.
- **V2 fails.** The calibration slope is **0.157** [−0.388, 0.922]: the model
  does not track cross-state variation in the observed part. V3 passes, but
  only for above-floor quantities.
- **What remains.** Without a distributional assumption, b(a) for a < F_app is
  bounded only by [0, b26(F_app)] ≈ [0, 0.85]. No P4 number below the floor
  should enter the welfare model, not even as a band. The only usable scenario
  is the assumption-free bound.

### 6. The M2 welfare feed (dw/da = 1 for all hours) is not justified by F1's estimates (major for the welfare model)

- **The feed is a mechanical step.** The registry's `interpretation_rule` maps
  P1 CI ∌ 0 ∋ 1 to M2 and then sets X_m = all hours. That mapping is the
  implementer applying the frozen rule, not an inference from the data. Under
  issues 1–4, the inputs to the rule are not robust:
  - Holm-adjusted P1 does not reject theta = 0.
  - The multistate-corrected P1 is indeterminate.
  - theta = 1 is rejected on the design frame (p = 0.0012), with the skill
    dose (p ≤ 3e−8) and by theta_T.
- **The difference is moderate in size.** At-floor hours (b26(F_app) = 0.851
  [0.802, 0.895]) have dw/da = 1 by construction for marginal floor changes.
  Above the floor, the estimates point to theta ≈ 0.35 (0.13–0.61). The
  implied exposure is x(F_app) = b + theta(1 − b) ≈ 0.90–0.94, not 1.0. M2
  overstates exposed hours by roughly 6–10%. That is small next to other
  uncertainties, but it is not "all hours". The registry also asks that a
  conflicting or indeterminate reading feed only the polar scenarios.
- **b26 is understated.** b26(F_app) counts 3.0% of M26 hours as unbound
  (r02f). These hours sit at a floor above F_app, mostly skill-I orders at the
  U.S. skill-II rate. Including them, the at-some-floor share is about 0.88.

### 7. Minor

- **G2.** The gap-premise correlation is 0.21 [−0.24, 0.55]. This agrees with
  A1 that a domestic-wage gap does not predict binding. It uses the same
  contaminated "above" classification, so read it as weak.
- **FALS1 passes** (placebo theta = 0.06 [−0.18, 0.30]). It checks state wage
  trends, not the classification or selection problems above.
- **FALS2 is uninformative** (0.38 [−0.67, 1.44]).
- **The INC selection invites regression to the mean.** Units are selected
  as wholly above in the base year (placebo mean dw = −0.09 while the floor
  rose 0.73). theta_T nets this out only if mean reversion is equal across
  transitions.

## Checks run (all post-outcome)

| Script | What it does | Key outputs |
| --- | --- | --- |
| `review/r01_p1_frames.R` | Reproduces P1 on the declared 285 and design 352 frames. Order-level selection check (no mixed units). Skill dose and unweighted variants on 352. Profile of the 67 dropped units. FY2026 position of INC units. LOSO influence. State means. Holm. | `r01_p1_frames.csv`, `r01_dropped_vs_kept.csv`, `r01_inc_fy2026_position.csv`, `r01_loso.csv`, `r01_state_means.csv`, `r01_holm.csv` |
| `review/r02_classification.R` | Hand recomputation of 40 orders from the source tables. Class and floor mix by experience months. Position of below-classes. Tolerance near-misses. Matches to any SOC rate (uninformative: the chance benchmark is higher). S3 shared values: roundness, top values, matches to prior AEWRs. At-floor-above-F_app mass. NY/OR areas. | `r02a_hand_check.csv` … `r02g_ny_or.csv` |
| `review/r03_p3_v1.R` | Counts behind P3. Delta_S under S3, stale-AEWR censoring, non-round S3, one-year S3, each at g and g = 0. M2-implied Delta_S under S3 (−0.0050). FY2026 position of units above a25 in FY2025. Absolute-hours (denominator-free) version. | `r03a_counts.csv`, `r03b_delta_s_variants.csv`, `r03c_above_a25_units.csv`, `r03d_absolute_hours.csv` |
| `review/r04_p4_coherence.R` | Refit of P4. Latent mass below the legal minimum. Implied wage levels against ACS. Upper-tail-only truncated fit. | `r04a_p4_coherence.csv`, `r04c_tail_only_fit.csv` |
| `review/r05_multistate.R` | Multi-state itinerary floors: share of "above" hours at another itinerary state's floor, chance benchmark, P1, P3 and theta_T after censoring, multi-state share of the 67 dropped units. | `r05_multistate_counts.csv`, `r05_chance_benchmark.csv`, `r05_multistate_p1_p3.csv` |

Logs: `$S/F1/review/r01.log` … `r05.log`. Run with
`R_LIBS_USER=… Rscript --vanilla <script> "$S"`, then
`git checkout -- Rplots.pdf`. I did not modify the registry, the implementer's
files, shared tools or producers.

## Use in the synthesis and the welfare model

**F1 can be used for these descriptive points (FY2026 field, hourly,
non-piece, 48 states, orders received from 2025-10-02 to 2026-06-30).**

- **Share of hours at a legal floor.** About 85–88% of certified hours are at
  some floor value: b26(F_app) = 0.851 [0.80, 0.90], plus about 3% at a floor
  above F_app.
- **Other hour shares.** 4.3% of hours are at the old FY2025 AEWR, a sticky
  wage that no model predicts. Between **6.0% and 7.7%** are strictly above
  every floor; the range depends on the multi-state correction.
- **The floor binds for most hours.** For the welfare model's X_m this
  supports a lower bound of x ≳ 0.85. Marginal floor changes reach at least
  that share of hours one-for-one.
- **Pass-through above the floor is partial.** theta_T = 0.39 [0.24, 0.56], or
  0.35 [0.15, 0.57] with the multistate correction, as a secondary,
  assumption-labeled number. The cross-state theta is 0.13–0.61 depending on
  frame, dose and classification. Pure M1 is contradicted by units that cut
  down to the new floor. Pure M2 (theta = 1) is rejected in most variants. A
  scenario range of dw/da above the floor of about [0.15, 0.6] is defensible;
  1 is the upper polar case.
- **Exposed-hours range.** Combining the two gives x(F_app) ≈ 0.90–0.94, with
  polar cases 0.85 and 1.0.
- **Measurement warnings for other strategies.** Two kinds of "above-AEWR"
  offers are floors in disguise: multi-state itinerary rates (43% of FY2025
  "above" hours) and stale prior-year AEWRs. Any design that uses an at-AEWR
  or binding flag must handle them. This adds to the A1 flag problem.

**F1 cannot be used for these points.**

- **Not as a recovered market or alternative-wage distribution.** P2 has no
  counterfactual-floor reading, because M1 is not supported. P4 is unusable
  below the floor: 50% of its latent mass lies below the minimum wage, the
  family drives it, and V2 fails. No P4 number, band or "scenario" should
  enter the welfare model. Below the floor, only the assumption-free bound
  b(a) ∈ [0, b26(F_app)] applies.
- **Not as the registry's M2 feed.** dw/da = 1 for all hours, with X_m = all
  hours, is the mechanical result of a non-robust P1 reading. If used at all,
  it is the upper polar scenario only.
- **Not as an elasticity.** F1 is not pooled and not converted, as the
  registry already states.
- **Not as a test of the gap-model premise beyond A1's finding.** G2 uses the
  contaminated classification.
