# A1 blind adversarial review (Phase 6)

- **Strategy:** A1, the 2025 statewide rule with revealed binding (D1). The
  dose is the within-state contrast b_c × (F_c − e25_s), FY2025 vs FY2026
  Q1–Q3, with county FE and state × FY FE.
- **Registry:** `explorations/identification_search/A1/registry.json`,
  SHA-256 `9a1182bd…0eed` (re-hashed by the reviewer; matches
  `registry.sha256`).
- **Reviewer:** a fresh Claude agent that did not build A1. Date: 2026-11-05.
- **Inputs used:** the registry; the code (`00_*`–`07_*`, `sources/`); the
  outputs in `$S/A1/`; the shared tools; the plan. Interpretive prose in
  `ledger_row.json` and in code comments was ignored.
- **Review checks:** all are **post-outcome** and not declared in the
  registry. Scripts are in `explorations/identification_search/A1/review/`;
  outputs are in `$S/A1/review/`.

## Verdict: **fails**

The implementation follows the registry, and every headline number reproduces
exactly. The design, however, does not identify the primary estimand:

- The registry's first stage is zero: π̂_U = 0.033 (SE 0.101), F = 0.10.
- The registry's own weak-first-stage rule therefore applies. The Fieller/AR
  set for ε_cost is the whole real line, and the jackknife SE of ε_cost is
  61.8.
- The review checks show why the first stage is zero. The FY2025 binding
  share b_c does not measure exposure to the cut. Half of the "not at AEWR"
  hours are non-field jobs whose own 2023-rule AEWR is missing from the
  schedule. Employers recorded as non-binding cut their offers by the same
  amount as binding ones.
- Because the dose moves no wages, the near-zero reduced form says nothing
  about the hours elasticity.
- The one significant secondary (SD5) is an artifact. Its declared-style
  placebo for FY2024→FY2025 gives the same coefficient (−0.0234, p = 0.029).

A1 should not enter the pooled estimate.

## Issues

### 1. The first stage is zero, and the zero is real, not a coding slip

- **Reproduction.** I rebuilt the declared first stage from the case rows and
  the dose file and got exactly the same result: π̂_U = 0.03269, CRV1 SE
  0.1012, F = 0.104 (`r01_r3_first_stages.csv`, R1).
- **Leave-one-state-out.** π̂_U ranges from −0.001 to 0.095
  (`a1_jackknife_primary.csv`).
- **The cut itself passed through.** On the same frame, the cross-state
  contrast (region × FY FE, raw effective cut) gives π = 0.61 (SE 0.056,
  F = 119; R3f). This matches Track 1's 0.66.
- **What fails is the within-state D1 contrast.** The implementer's
  post-outcome descriptive shows the same offered-wage change in the high-
  and low-b halves of each state: −$2.38 and −$2.37, against predicted
  changes of −$3.44 and −$3.11.
- **Weights and composition.** The first stage is also fragile to both.
  - Unweighted, π = −0.36 with F = 13: a strong first stage of the wrong
    sign (R3e).
  - With field-SOC rows only in ω, π = 0.32 with F = 8.4 (R3c).
  - No variant reaches F ≥ 10 with the expected sign.

### 2. The binding share b_c is mismeasured

**Non-field jobs (decisive).**
- The shared at-AEWR flag (`shared/01_outcomes.R`, `at_aewr_rows`) compares
  FY2025 offers only with the field/livestock reference AEWR. The
  `aewr_rate_schedule.parquet` file has only the `nonrange_reference` and
  `fls_common_soc_reference` scopes.
- Under the 2023 rule (in force from 2023-03-30;
  `markdowns/aewr_2023_rule_change_design.md`), non-field SOCs had
  occupation-specific OEWS AEWRs. Those rates are not in the repository.
- Among FY2025 rq13d known-AEWR hourly hours (`r01_r2_*`):
  - **Non-field SOCs hold 3.5% of hours but 50.4% of the "not at AEWR"
    hours.** Their at-AEWR share is 0.145, and their median wage is 1.26 ×
    the field AEWR.
  - The leading SOCs are 53-3032 (40%), 47-2061 (22%) and 53-3053 (12%).
- Non-field offers are not unconstrained. **86% of their hours sit at the
  modal wage for their state × SOC cell**, against 77% for field jobs
  (`r03_nonfield_mass_points.csv`). That is the signature of a binding floor
  the flag does not know about.
- Their FY2026 offers fell by **$2.04**, against $2.31 for field jobs
  (`r03_wage_change_field_nonfield.csv`). The A1 dose instead gives them
  weight (1 − b) ≈ 0, as if they were unmoved.
- Consequences:
  - b_c correlates −0.58 with the county non-field share (`r01_r3_*`).
  - This is the registry's own F6 balance failure: the dose's within-state
    correlation with the FY2025 field-SOC share is −0.59 (p = 5×10⁻⁷).

**Field jobs.**
- Of the field hours flagged "not at AEWR", **55% are more than $0.05 below
  the FY2025 AEWR** (`r01_r2c_*`). Louisiana alone is 29% of these, with a
  median gap of −$0.21 (`r01_r2e_*`).
- That points to schedule or reporting mismatches, not offers above the
  floor.
- Only about 13% of all not-at-AEWR hours are field hours more than $0.50
  above the AEWR. Those are the hours the D1 logic relies on.
- I did not resolve the cause. It is a shared-tool issue that also affects
  any D1 layer on A2 or A3.

**Employer-level check (field-only incumbents).**
- Employers with b_e ≤ 0.25 cut offers by **$2.39**. Employers with
  b_e = 1 cut by **$2.32** (`r01_r4d_*`).
- The employer × state first stage is −0.13 (F = 4.9) overall and −0.006
  (F = 0.005) for field-only employers (`r01_r4_*`).
- So even correctly measured "above-floor" field employers kept their
  premium and passed the cut through. **Assumption A-binding is false in the
  data.**

### 3. The conversion is uninformative, and the ITT "bound" is not a bound

- **Fieller/AR set.** I recomputed it from the leave-one-state-out
  covariance. a = π² − t²V₂₂ = −0.043 < 0 and the discriminant is negative,
  so the set is the whole line. The point ε_cost = −2.61 (jackknife SE 61.8)
  is meaningless.
- **The ITT at π = 1.** It gives ε_itt = −0.085 [−0.54, 0.37]. That is an
  elasticity only if π = 1, which the data reject (t = (1 − 0.033)/0.101 ≈
  9.6).
- **The interval [ε_itt, ε_cost] = [−2.61, −0.085].** It assumes
  0 < π ≤ 1 is known. The π CI is about [−0.17, 0.24], so the interval is
  not an identified set.
- **The reduced form.** η = −0.0045 (p = 0.71) is what a zero first stage
  predicts whatever the true elasticity. It is not evidence of a small
  elasticity.

### 4. The design power was computed at π = 1

- Design MDE: 0.82 under the gate DGP. It was flagged above 0.5 and
  "conversion-sensitive" (0.82/0.65 = 1.26).
- The realized MDE in ε_cost is 19 (CRV1 at the point π) or 173 (jackknife
  ratio). The registry's pooling eligibility at 0.82 rested on π = 1.
- Inference rule:
  - It was applied as declared: the node ratio of 1.64 > 1.5 sends the gate
    to the shared DGPs, where state CRV1 passes.
  - Under county_serial, state CRV1 (rejection 0.065, Wilson upper 0.082)
    and state CV3J (0.060 / 0.077) fail the size gate.
  - This is minor here because every p-value is far from 0.05.
- Using t(G − 1) critical values is consistent with the harness that
  certified size (`power_harness.R` uses G − 1 df). It is not a deviation.

### 5. The SD5 employer × state result (η = −0.0221, p = 0.0056) is an artifact

From `r02_sd5_robustness.csv`:

| Check | η | p (state CRV1) |
| --- | --- | --- |
| Reproduction | −0.0221 | 0.0056 |
| **Placebo FY2024→FY2025** (b from FY2024, same cut, FY2024 incumbents) | **−0.0234** | **0.029** (jackknife 0.031) |
| Intensive margin (units positive in both years) | +0.0012 | 0.83 |
| Control: FY2025 non-field share × post | −0.0087 | 0.27 |
| Field-only units | −0.0141 | 0.030 (jackknife 0.067) |

- The whole effect comes from hours-weighted exit.
  - FY2025 incumbents with b_e ≤ 0.25 are 71% non-field. 25% of their hours
    exit, against 13% for b_e = 1 (`r02_q7_exit_by_b.csv`).
  - The same gradient appears in a year with no policy change.
- Its employer-level first stage is −0.13 (issue 2), so SD5 cannot be
  converted either.
- **Multiplicity.** Holm across the 24 non-primary A1 estimate rows gives a
  minimum adjusted p of 0.13. Two rows have unadjusted p < 0.05 (SD5, and
  the equal-county ratio at p = 0.036).
- **Employer linkage.** This is not the cause. Hours without an
  employer_id_balanced are 0.22%, 0.83% and 0.19% in FY2024, FY2025 and
  FY2026. The FY2020 linkage break lies outside the FY2024–26 frames.
- The leave-one-state-out range is −0.027 to −0.021, so no single state
  drives it. The artifact is compositional, not a leverage problem.

### 6. Sign flips in sensitivities

- S2 (b from FY2023–24): +0.024 (p = 0.24). S9 (field-share and weeks
  controls): +0.021 (p = 0.20).
- With π ≈ 0 these are noise around zero.
- S9's flip confirms that the primary reduced form loads on the field-share
  imbalance in issue 2.

### 7. Integrity: no violation found

- The registry was frozen at 19:39 (file times).
  - `01_support` outputs (19:35) are pre-freeze treatment and support
    statistics, which the plan allows. The code reads outcomes only for
    years ≤ 2025 and FY2025 case rows.
  - Power ran at 19:43, falsification at 19:52, estimation at 19:54.
- The only undeclared output is `07_post_outcome_first_stage_descriptive`,
  and it is labeled post-outcome.
- Choices the registry left open, all consistent with it:
  - **SOC source.** F5(c) and S7 name the source. Case rows lack SOC for
    59.3% of FY2025 rq13d hours (verified), so disclosure-record SOC is the
    only workable source.
  - **Covariate imputation.** It reuses the registry's b_c rule.
  - **t(G − 1) p-values.** These match the harness.
- **One declared alternate was not run:** SD2 RI by state within region. It
  is disclosed in the output note (F_c varies within NY and OR). This is
  immaterial.

### 8. Data and construction checks that passed

- **Minimum-wage table.** I spot-checked it against known 2025–26 values:
  CA 16.50/16.90, WA 16.66/17.13, AZ 15.15, CO 15.16, CT 16.94, FL
  14 (2025-09-30), MI 13.73, MO 15, NJ agricultural 14.20, NY 16/17,
  OR 15.05/14.05, RI 16, MN 11.41. Farm exclusions (PA, VA, VT, NE, DE)
  are coded as 7.25.
- **Maine.** The 2026 coverage of farmworkers ($15.10) was not
  independently verified. Within-state, it only scales Maine's slope.
- **Doses.** e25 = AEWR on 2025-04-01 (e.g., CA 19.97). The skill-I rates
  are from the 2025-10-02 sheet. The minimum wage never reaches the FY2025
  AEWR, which the code checks.
- **Receipt windows.** FY2026 rq13d covers receipts from 2025-10-01; the
  first stage uses receipts from 2025-10-02. Both are correct.
- **PPML.** fixest::fepois with county + state × FY FE. S10's rebuild of the
  outcome matches the shared panel (max difference < 1e−4).
- **Falsification gates pass:**
  - F1 placebo: p = 0.27.
  - F2 drift: p = 0.37.
  - F3 joint jackknife Wald: p = 0.86.

## Checks run (all post-outcome)

| Script | What it checks |
| --- | --- |
| `review/r01_binding_share_and_first_stage.R` | Reproduces π_U, η and F. Breaks down FY2025 not-at-AEWR hours by field vs non-field SOC and by gap to the AEWR. Computes a field-only binding share. Re-estimates the first stage with different ω compositions, weights and FE. Runs the employer × state first stage and wage change by b bin. Runs the primary reduced form with the field-only b (η = +0.0024, p = 0.86). |
| `review/r02_sd5_robustness.R` | SD5 reproduction, b-known units only, intensive margin, exit LPM, non-field control, field-only units, FY2024→FY2025 employer placebo, linkage shares, exit by b bin, leave-one-state-out. |
| `review/r03_nonfield_mass_points.R` | State × SOC modal-wage bunching for non-field vs field jobs, and the FY2025→FY2026 offered-wage change by group. |

Logs and CSVs are in `$S/A1/review/` (`r01.log`, `r02.log`, `r03.log`,
`r01_*.csv`, `r02_*.csv`, `r03_*.csv`).

## Use in the synthesis

**What A1 can be used for:**
- As a documented negative design result: within a state, the FY2025
  at-AEWR share does not predict which county's (or employer's) offered
  wages fell under the 2025 rule (π̂ = 0.03, F = 0.10).
- The 2025 cut passed through broadly and nearly uniformly. Field offers
  fell about $2.31 and non-field about $2.04. Above-floor employers kept
  their premium, which argues against D1 revealed-binding layers generally.
- As a warning about the shared at-AEWR flag. It omits the 2023-rule
  occupation AEWRs, and about half of FY2025 field "non-binding" hours lie
  below the AEWR. Any strategy that uses `share_at_aewr` (D1 on A2/A3, bite
  measures) should be re-examined.
- The A1 runs confirm cross-state pass-through of 0.61 on rq13d (R3f,
  post-outcome), consistent with Track 1.

**What A1 cannot be used for:**
- Any point estimate, bound or CI for ε_cost. That includes ε_itt = −0.085
  and the [−2.61, −0.085] interval.
- Evidence that the hours elasticity is small or zero.
- SD5 as evidence of an incumbent response.

**Pooling eligibility: not eligible.**

| Criterion (`declarations.json` `pooling_rule.inclusion`) | Result |
| --- | --- |
| (1) Frozen registry | Pass |
| (2) Design MDE ≤ 1.0 | Pass, at 0.82, but only under the π = 1 premise the data reject |
| (3) Falsification gates | Pass |
| (4) Review verdict | **Fails** |
| (5) Converts to ε_cost | **Fails**: Fieller/AR set is the whole line; size-valid jackknife SE 61.8 |
| (6) Distinct assumptions | A1 family counts once with Track 1 |

- If a synthesis pooled A1 anyway, its weight at SE 61.8 would be
  negligible.
- Pooling the ITT (SE about 0.22) would wrongly impose π = 1 and must not
  be done.
- A1's primary hypothesis still belongs in the Romano–Wolf/Holm family,
  because it reached estimation. Its size-valid p-value is 0.71 (CRV1) or
  0.72 (AR/jackknife).
