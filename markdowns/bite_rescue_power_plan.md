# Investigation Plan: Rescuing a Higher-Power Bite-Based Design on Cached Local Data

> **Purpose:** Self-contained research and execution plan designed to be delegated to a fresh **Gemini Neon** instance operating in `/home/ken/Dropbox/projects/H-2A_Paper`.
> **Core question:** Can *any* economically valid, size-valid bite-based design built strictly from data already cached in this repository achieve materially higher statistical power than the existing benchmarks — improving cleanly on the current county-level minimum detectable elasticity (MDE) floor without cheating inference, violating repository contracts, or p-hacking?

---

## 0. Mandatory Grounding & Hard Guardrails (Read First)

Before writing a single line of code, the delegated instance **must**:

1. Run from the repository root:
   ```sh
   git status --short
   python scripts/agent_docs.py verify
   python scripts/agent_docs.py snapshot --scope code/designs/bite
   python scripts/agent_docs.py snapshot --scope code/designs/policy_2025
   ```
2. Read `AGENTS.md`, `content/designs/bite.md`, `content/designs/policy-2025.md`, `markdowns/identification_search_report.md`, and `outputs/reports/bite_design.md`.
3. Obey all hard repository and user constraints:
   - **Isolation:** Write **all** new exploration code and intermediate outputs strictly under `explorations/bite_power_rescue/` and write the final report to `markdowns/bite_rescue_power_report.md`. **Do NOT edit** `code/c01_clean/`, `code/c02_build/`, `code/designs/*`, `code/welfare/`, `scripts/*`, `draft/*`, `content/*`, `agent/assumptions.toml`, or any retained `outputs/*` artifacts. Promotion of any winning design is a separate user decision.
   - **Data boundaries:** Use **cached local data only**. Non-public NASS data is unavailable (`ACCESS-001`). Do **not** search external legal dockets (e.g., *UFW v. DOL*), run credentialed API pulls, or assume waiting for `2026Q2` QCEW/QWI or `FY2027` disclosures.
   - **Data integrity:** Keep all geographic identifiers as canonical strings (`GEO-001`); keep canonical `2009` statutory `aewr` strictly `NULL` (`AEWR-001`); use the unified shared panel (`data/processed/county_year_panel.parquet`, SHA-256 `4e2e56d0...`) and unified `data/intermediate/state_real_minwages.parquet` (`14fe913b...`, **never** retired `fred_state_minwages.parquet`).
   - **Code style:** Imperative R (`Rscript --vanilla`) or imperative Python/Marimo (`.py`, zero classes), run Python via `UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync`.
   - **No cheating inference:** Never downgrade clustering from the level of treatment assignment (`aewr_region_id` $G = 17$ for FLS AEWR; `state_fips` $G = 46$ for the Oct 2, 2025 statewide rule) or substitute `CRV1` / county / CZ SEs for exact delete-one-cluster jackknife (**CV3J**, $t(G-1)$) or valid randomization inference (RI).

---

## 1. Ground Truth: What Is Already Dead (Do Not Re-Run)

The delegated agent must **not** spend time re-testing these already-falsified specifications:

| Already Tested & Killed | Repo Location | Exact Failure Mode & Realized MDE |
|---|---|---|
| **Original DiD (`aewr_cz_p25_l1 × post`)** | `code/designs/did/` | Drops 741 "always takers" (`62.6%` of US H-2A workers); `88.1%` residual variance is endogenous `wage_p25_l1` (wrong sign `-0.012`); fake `2011` break; severely under-clustered `cz × region` SEs; wrong-sign crop price & labor share. |
| **Full-panel FE-PPML frozen wage-gap bite (`E_p25`, `E_oews`, `E_sf_log × ln(AEWR)`, `2011–2025`)** | `code/designs/bite/` | Clean specification, but **`1b-PPML` hours `+0.003 [CV3J 0.135]`** (`95% CI [-0.28, +0.29]` per SD → **`[-0.57, +0.58]` across a 2-SD spread**, MDE `0.38` per SD / `0.76` per 2-SD), leaning `96.6%` on `E_c × Year` trends; **`1a-PPML` (`+ E_c × Year`) explodes to MDE `2.11`**. State-year deviance variance is `2.85×` county noise; `G_eff = 13.36` regions. |
| **QWI county farm-earnings exposure (`E_qwi`)** | Prior audit (`code/designs/bite/` prep) | Loses `509` counties (`2,399` covered), `89.7%` negative raw gaps, fails `2008–10` pre-trends (`p = 0.013`) and lead placebo (`+0.60`). |
| **LOO simulated nonlinear bite (`max(0, ln(AEWR) - ln(w_sim))`, `AEWR / w_sim`)** | Prior audit (`code/designs/bite/` prep) | Ends `2024`; `R²(C+RY) = 0.9935–0.9995`; weak/wrong-sign first stages; `CV3J` MDE `6.1–27.9`. |
| **Standalone `ln(prevailing_ag_min_wage)` translated to AEWR** | `code/designs/bite/` | Direct farm-wage pass-through is only `λ = 0.047–0.063` in QWI, inflating translated AEWR MDE to `7.5–23.0`. |
| **Strategy A1 (within-state at-AEWR share × Oct 2025 cut)** | `markdowns/identification_search_report.md` §2.2 | Failed first stage (`π̂ = 0.033, F = 0.10`): unbound incumbents cut offers as much as bound ones (`$2.39` vs `$2.32`); non-field `2023` AEWRs misclassified binding share. |
| **Other pre-2025 county panels (`Panel IV`, `Border v7`, `MCW v4.1.0`) & killed search candidates (`A2` standalone, `A3`, `N1`, `A4`, `C1–C4`, `F1`)** | `code/designs/{panel_iv,border_discontinuity,mundlak_chamberlain}/`, `identification_search_report.md` | `Panel IV` hours MDE `~16.8`; `Border v7` `0/8` precision gates; `MCW` max leverage `0.97` (`17` regions); `A2` MDE `2.78–6.95`; `A3` MDE `>5.5`; `N1` MDE `4.35` at `π = 0.138`. |

### Current Benchmarks to Beat
Any rescued bite design is only useful if it **materially improves on** what the repository already has in production:
- **Aggregate `FY2026 Q4` Track 1 PPML certified hours (`code/designs/policy_2025/`):** `-0.078 [-0.598, +0.443]`, **MDE `0.72`** (`G = 46` states); direct `ln(rate)` **MDE `0.54`**; positions **MDE `0.52`**; `ε_cost` **MDE `1.51`**.
- **Direct-grower (`65.9%` of hours) `FY2026 Q4` PPML (`11b_estimate_mechanisms.R`):** hours `-0.066 [-0.440, +0.308]`, **MDE `0.52`**; positions `-0.021 [-0.293, +0.251]`, **MDE `0.38`**.
- **Full-panel `2011–2025` FE-PPML bite (`code/designs/bite/`):** hours **MDE `0.38` per SD** (`0.76` across a 2-SD contrast) in `1b-PPML`; **MDE `2.11`** in `1a-PPML`.

---

## 2. The Statistical-Power Diagnosis (Why SEs Were Wide)

In a clustered bite regression with effective clusters $G_{\text{eff}}$, residual outcome noise $\sigma_\varepsilon$, residualized treatment dispersion $\text{SD}(\tilde{X})$, and wage pass-through $\pi$:

$$\text{MDE}_{\text{cost}} \;\propto\; \frac{\sigma_\varepsilon}{\sqrt{G_{\text{eff}}} \cdot \text{SD}(\tilde{X}) \cdot \pi}$$

Previous bite designs struggled because:
1. **Numerator ($\sigma_\varepsilon$) is inflated by state-year demand shocks (`2.1×–2.85×` pure county noise) and H-2ALC contractor reallocation noise** (H-2ALCs are `34.1%` of hours yet have `6.2×` the jackknife SE of direct growers: `1.148` vs `0.186`!).
2. **Effective clusters ($G_{\text{eff}}$) are capped at `13.4` regions** on `2011–2024` FLS variation (vs `46` states on the Oct 2, 2025 rule).
3. **Local-wage exposure (`E_p25`) has almost zero pass-through bite ($\pi_{\text{diff}} \approx 0$)** because `93%–97%` of H-2A hours sit directly at the AEWR regardless of county `wage_p25`.
4. **Adding `E_c × Year FE` destroys $\text{SD}(\tilde{X})$** (`0.150 → 0.028`, collapsing surviving variance by `96.6%`).

Therefore, **only** designs that attack at least one of **$\sigma_\varepsilon$**, **$G_{\text{eff}}$**, **$\text{SD}(\tilde{X})$**, or **$\pi$** *without sacrificing identification* have any chance of rescuing bite.

---

## 3. Strict Two-Phase Outcome-Blind Protocol (Relaxed Power Gate, Strict Credibility)

To eliminate specification search and p-hacking while avoiding premature dismissal of promising improvements, enforce a hard two-phase gate with **strict size/placebo discipline** and **relaxed MDE promotion cutoffs**:

### Phase 1: Strictly Outcome-Blind Power, First-Stage & Size Triage
**Rule:** During Phase 1, **DO NOT estimate any `FY2026` H-2A quantity outcome contrast** for new specifications (and for `2011–2025` bite candidates, calibrate power strictly on **FE-only pre-period / placebo-rotated residuals** before estimating the treatment slope!).

For each candidate **R1–R6** below, compute and record in `explorations/bite_power_rescue/phase1_triage.csv`:
1. **Exact estimand label**: Aggregate elasticity ($\eta$), differential bite elasticity ($\theta$, specifying exact 1-SD and 2-SD economic meaning), or within-firm reallocation elasticity ($\sigma_{\text{realloc}}$).
2. **Sample coverage**: `N` units, `N` counties/businesses, share of total US H-2A hours covered, and `G_eff` (exact Fisher-information effective clusters).
3. **Regressor survival**: `R²` and residual `SD(X̃)` after proposed FEs (and after `+ E × Year FE` where applicable).
4. **First-stage wage / cost-bite verification** (where bite claims differential wage pass-through): $\hat{\pi}$, CV3J SE, $F$-stat. **Kill if $F < 10$ or $|\hat{\pi}| < 0.25$** (filters out dead first stages like A1's `F = 0.10`, while preserving moderate genuine pass-through).
5. **Empirical shared-shock ratio**: $\text{MS}_{\text{state} \times t} / \text{MS}_{\text{unit}}$ on pre-policy FE-only residuals.
6. **Size-validated Design MDE (500–999 placebo draws)**:
   - Draw placebo policy assignments (permuting state doses within former AEWR regions for `2025` designs, or permuting regional AEWR paths across the 17 regions for `2011–2025` designs) on **actual pre-policy outcomes** (`FY2012–FY2025` for `2025` designs; `2008–2010` / shifted-region draws for `2011–2025`).
   - Verify **empirical size $\in [0.03, 0.075]$** at nominal $\alpha = 0.05$ under exact `CV3J` (or RI).
   - Compute the **size-valid 80%-power MDE** in economically comparable elasticity units (converting differential $\theta$ to a **2-SD high-vs-low bite spread** AND reporting implied $\varepsilon_{\text{cost}}$!).

### Relaxed Phase 1 → Phase 2 Promotion Gate
A candidate advances to **Phase 2 (Outcome Estimation)** if it satisfies **all four** criteria:
1. **Size validity (non-negotiable):** Empirical rejection rate $\in [0.03, 0.075]$ at $\alpha = 0.05$ under actual empirical error correlation (never iid noise).
2. **Material power improvement (relaxed cutoffs):**
   - **Aggregate / effective-floor** elasticity candidates (`R1`, `R2-2025`, `R6`): size-valid statutory hours **MDE $\le 0.60$** (or $\ge 15\%$ tighter than `policy_2025` `full_fy` `0.72`) AND **$\varepsilon_{\text{cost}}$ MDE $\le 1.20$** (vs current `1.51`).
   - **Direct-grower / firm-cohort / reallocation** candidates (`R2`, `R3`): size-valid hours **MDE $\le 0.45$** (or $\ge 15\%$ tighter than `direct_grower_hours` `0.52`; positions **MDE $\le 0.35$** vs `0.38`).
   - **Differential-bite** candidates (`R4`, `R5`, `R2-bite`): size-valid **2-SD spread MDE $\le 0.60$** (`1-SD MDE <= 0.30`, $\ge 20\%$ tighter than `code/designs/bite/`'s `0.38` / `0.76`) — and for `1a-PPML` (`+ E_c × Year FE`), any design reaching **MDE $\le 1.00$** (vs current `2.11`).
3. **Clean pre-policy placebos & balance:** Passes `FY2024–FY2025` (or `2008–2010`) placebo (`p > 0.10`; flag `0.05 < p <= 0.10` as conditional with explicit caution rather than silently passing) and baseline covariate balance.
4. **Economic interpretability:** Clear mapping to labor-demand elasticity, within-firm cross-state substitution, or cost-share incidence.

Classify Phase 2 survivors into two transparent tiers:
- **Tier 1 (High-Power Promotable):** Size-valid hours MDE $\le 0.40$ (or 2-SD spread $\le 0.45$).
- **Tier 2 (Moderate-Power Improvement):** Clears the relaxed gate (`0.40 < MDE <= 0.60`) and materially beats current production benchmarks.

> **Stop Rule:** If **zero** candidates clear even the relaxed Phase 1 gate, **STOP immediately without estimating noisy Phase 2 outcome regressions**, and publish a clean negative-result power proof in `markdowns/bite_rescue_power_report.md`.

---

## 4. Six Concrete Unexplored Rescue Candidates to Audit in Phase 1

### Candidate R1: Effective Competing-Floor Truncation Bite on the Oct 2, 2025 Cut (`46` States + Sub-State Tiers)
- **Economic motivation:** Before Oct 2, 2025, state agricultural minimum wages exceeded the AEWR on **`0.0%`** of H-2A hours. After the `-$3.74/hr` Oct 2, 2025 cut, **`17.61%`** of `FY2026` H-2A hours sit in states where `prevailing_ag_min_wage > AEWR_2026` (`outputs/reports/policy_2025.md:61`). Track 1 currently regresses on the **untruncated** statutory `x_us_skill_1`, introducing attenuation and noise wherever state or sub-state ag minimum wages prevent offers from falling all the way to the new AEWR!
- **Exact sub-designs to test in Phase 1:**
  - **R1a (Truncated Effective Statutory Floor):** Replace raw `x_us_skill_1` with $x^{\text{eff}}_{s,t} = \max(\text{AEWR}^{\text{US-I}}_{s,t},\; \text{prevailing\_ag\_min\_wage}_{s,t})$ using unified `data/intermediate/state_real_minwages.parquet` (46 multi-state states, all complete through `2026`!). Test whether first-stage offered-wage pass-through $\hat{\pi}$ jumps from `0.623`/`0.671` toward `0.75–0.90` and how much `ε_cost` MDE drops!
  - **R1b (Pure Truncation Bite):** Interact or enter separately `truncation_gap = max(0, ag_min_2026 - aewr_2026)` alongside `raw_cut` — isolating states where the legal minimum-wage floor prevented the AEWR cut from binding!
  - **R1c (Within-State Sub-State Minimum-Wage Tiers):** Using `explorations/identification_search/A1/sources/minwage_substate_counties.csv` (NY downstate vs upstate, OR Metro/Standard/Nonurban tiers, plus CA/WA/AZ local ag floors), check whether **within-state** sub-state truncation variation has enough $G_{\text{eff}}$ and $\text{SD}(\tilde{X})$ after `State × FY FE` (which eliminates the `2.1×` state-year shared shock!) — kill immediately in Phase 1 if only 2–3 states drive sub-state variation.

### Candidate R2: Direct-Grower Restriction + Variance-Absorbing Covariates & Factor Models (`2011–2025` Bite AND `2025` Rule)
- **Economic motivation:** Move 1 proved H-2ALCs (`34.1%` of hours) have **`6.2×` larger jackknife SEs** than fixed-site direct growers (`1.148` vs `0.186`), and `2.1×–2.85×` state-year residual variance is driven by unobserved crop/weather/regional demand shocks.
- **Exact sub-designs to test strictly on pre-policy FE-only residuals in Phase 1:**
  - **R2a (Direct-Grower-Only `2011–2025` Bite):** Re-evaluate `code/designs/bite/` restricting outcomes to `h2a_labor_contractor == "N"` (`65.9%` of hours) — measure exact change in `MS_SY / MS_county`, `G_eff`, and placebo `CV3J` MDE (`1b` and `1a`).
  - **R2b (Exogenous Weather & Crop-Mix Shock Absorption):** Test adding strictly predetermined/exogenous agricultural controls already cached in this repo to shrink $\sigma_\varepsilon$ without absorbing statutory AEWR:
    - County baseline CDL crop shares (`croplandcros_county_crop_type_acres.parquet`: fruit/tree nuts, vegetables, field crops, horticulture) $\times$ Year FE (absorbs national crop-specific demand/price booms!).
    - County NOAA annual climate anomalies (`data/intermediate/` NOAA artifacts from `code/a01_sources/09_01_h2a_prediction_pull_noaa.py`) and county `fisher_index_ppi` / `fisher_quantity_index` (keep NOAA weather + predetermined `CDL_0810 × Year FE` primary and `fisher` strictly sensitivity since crop prices can respond to supply).
  - **R2c (Lagged-Outcome ANCOVA, Synthetic DiD [SDID], & Interactive Fixed Effects [IFE]):** On `FY2012–FY2025` (for `2025` rule) and `2011–2024` (for `2011–2025` bite), test whether conditioning on pre-period outcome trajectory (`ln(H_{c,t-1})`, SDID unit+time weights, or 1–2 factor IFE / matrix completion) cuts the `2.1×–2.85×` state-year shared-shock ratio AND **preserves empirical size $\le 0.075$** under placebo state/region permutations! *(Critical referee check: IFE/SDID can easily over-fit with $G=17$ or $46$; Phase 1 placebo size validation is mandatory!)*

### Candidate R3: Fixed-Cohort Worksite-Business Panel & Within-Multi-State-Employer Reallocation
- **Economic motivation:** County aggregates mix firm entry, firm exit, multi-county worksite splitting, and H-2ALC churn. Moving to the **firm/worksite-business level** removes firm-specific crop, housing-capacity, and management shocks via firm fixed effects $\alpha_f$ — and for multi-state employers, `filer_id × FY FE` removes **all** firm-wide shocks!
- **Exact sub-designs to test in Phase 1:**
  - **R3a (Balanced Fixed-Cohort Direct-Grower Business Panel, `FY2020–FY2026`):** Using canonical post-FY2020 `employer_id_balanced` (`employer_is_worksite_business == TRUE`, zero FY2020 linkage break!) restricted to direct growers (`h2a_labor_contractor == "N"`), fix a baseline cohort (e.g., businesses active in `FY2023–FY2024`, keeping explicit `Y = 0` for exits so survivor bias is impossible!), and test FE-PPML with `business FE + Former-Region × FY FE` (state CV3J) on `FY2025–FY2026` (calibrated strictly on `FY2024–FY2025` & earlier placebo cohorts!).
  - **R3b (Within-Multi-State-Employer Cross-State Reallocation, `FY2025–FY2026`):** In `outputs/reports/policy_2025.md:13`, `within_filer_across_state` already has **`2,024` continuing filer–state–occupation cells across `46` states** and a clean offered-wage first stage (`$0.442 [0.205, 0.679]`). Construct the **balanced multi-state `R7` filer × state × FY panel** (retaining zeros for states where a multi-state filer was active at baseline!) and run FE-PPML with **`filer × FY FE + filer × state FE`** (state-deletion CV3J, plus two-way `state + filer` check).
    - **Estimand discipline:** Explicitly label R3b as the **within-firm cross-state substitution / reallocation elasticity** (NOT the aggregate market-level elasticity!), and check in Phase 1 whether absorbing `filer × FY FE` slashes `MS_SY / MS_unit` and delivers MDE $\le 0.45$ with clean `FY2024–FY2025` placebos!

### Candidate R4: Structural Cost-Share Bite (Contract-Duration Amortization & Farm Labor-Intensity Bite)
- **Economic motivation:** Even when 95% of H-2A jobs sit at the AEWR, a `10%` AEWR change does **not** raise total hourly H-2A labor cost uniformly across employers or counties:
  1. **Fixed-cost amortization bite:** Non-wage H-2A fixed costs $F$ (inbound/outbound transport, visa/filing, housing setup) are largely invariant to contract length $H_{\text{weeks}}$. Thus $\frac{\partial \ln C}{\partial \ln \text{AEWR}} = \frac{w \cdot H}{w \cdot H + F}$ is **strictly higher for long-duration contracts** (35–42 weeks, ~85–90% cash-wage share) than short-duration contracts (8–12 weeks, ~55–65% cash-wage share).
  2. **Farm production labor-cost-share bite:** By Marshall's third law of derived demand (holding output elasticity constant), a `1%` wage shock shifts labor demand proportionally to the labor cost share $s_L$ (`share_farm_laborexp_prodexp` `2008–2010` or baseline specialty-crop intensity).
- **Phase 1 test:** Construct frozen predetermined `2008–2010` (for `2011–2025` bite) and `FY2022–FY2024` (for `2025` rule) county/firm **duration-amortization share** and **farm labor-cost share** exposures; verify `SD(X̃)`, `G_eff`, placebo size, and MDE under `County + Region×Year` and `County + State×Year` FE (plus direct-grower restriction).

### Candidate R5: Local-Wage Crossover Bite on the Oct 2, 2025 Cut
- **Economic motivation:** On `2011–2024`, AEWR stayed well above local farm/low-wage benchmarks almost everywhere. The `-$3.74/hr` Oct 2, 2025 cut created a genuine **crossover regime**: in some counties the new statutory AEWR fell *below* local nonfarm/OEWS/QWI prevailing wages (`AEWR_2025 > w_local > AEWR_2026`), where market competition may bind offers above the new AEWR (reducing wage pass-through $\pi$!), while in low-wage counties `AEWR_2026` remained well above local wages (full pass-through!).
- **Phase 1 test (First-Stage Gate FIRST!):** Before touching quantity outcomes, test directly on `policy_2025`'s `FY2025–FY2026` offered-wage panel (`wage_filer_state_occupation.parquet`) whether predetermined `2024` county `OEWS Big-Six` / `QWI NAICS 111` / `ACS p25` crossover indicators predict **statistically distinct offered-wage pass-through $\Delta\hat{\pi}$** (`F >= 10`, $|\Delta\hat{\pi}| \ge 0.25$, clean `2024–2025` wage placebo)!
  - **Immediate kill if $\Delta\hat{\pi}$ fails `F >= 10`** (confirming market wages still do not create differential H-2A wage bite even after a `-$3.74/hr` cut, just like A1!).
  - If $\Delta\hat{\pi}$ passes, proceed to pre-outcome placebo quantity power calibration under `State × FY FE` (which eliminates the `2.1×` state-year shock!).

### Candidate R6: Stacked Multi-Event Statutory-Reversal Design (`A6`: `2009–2010` OES-to-FLS + `2025` Statewide Cut)
- **Economic motivation:** `A6` was explicitly left **not run** in `markdowns/identification_search_report.md:112`, even though `explorations/identification_search/sources/aewr_2009_oes_schedule.csv`, `oes_area_county_crosswalk_2009.csv`, and `h2a_disclosures_fy2006_2007.csv` are already cached locally! Stacking an **upward** reversal (`2009 → 2010`, return from sub-state OES to regional FLS) with a **downward** cut (`2025 → 2026`, shift from regional FLS to statewide OEWS) across **two opposite-signed statutory episodes** adds independent cross-state/sub-state policy shocks and directly counters unidirectional state-trend confounding.
- **Phase 1 test:**
  1. First verify `2009 → 2010` **offered-wage first stage** (`π_2010`, `F_2010`) using the cached `aewr_2009_oes_schedule.csv` (explicitly handling the `64%` minimum-wage truncation in `2008–09`, missing `2008–10` weekly hours [so test **certified positions**, NOT imputed hours!], and `AEWR-001` — never using `aewr_nonstatutory_application_wage_2009` as statutory dose!).
  2. Test pooled stacked `event × county` + `event × former_region × period` FE with cluster-robust `state` CV3J across the 2 stacked episodes (`46` states × `2` events): compute exact placebo size and pooled **positions** MDE vs `policy_2025`'s single-event `0.52` (`direct_grower` `0.38`)!
  3. **Kill rule:** If `2009–10` first stage fails `F >= 10`, or if `2009–10`'s low dose SD (`$0.14` within-state) contributes `< 15%` of total Fisher information to the stacked estimator (meaning the stacked estimate is just `policy_2025` with extra noise), kill R6 cleanly in Phase 1!

---

## 5. Phase 2 Execution (Only for Phase 1 Survivors)

For **only** the candidates that clear all four Phase 1 gates:

1. Write an explicit frozen registry `explorations/bite_power_rescue/phase2_registry.json` + SHA-256 lock **before** running post-policy outcome estimation.
2. Estimate the registered specifications alongside:
   - Exact matched pre-policy placebos (`FY2024–FY2025` and/or `2008–2010`),
   - Exact delete-one-cluster jackknife (`CV3J`, $t(G-1)$) + leave-one-cluster-out range + Fisher-info $G_{\text{eff}}$,
   - Holm step-down multiplicity-adjusted $p$-values across all promoted Phase 2 tests,
   - Explicit `ε_cost` conversion accounting for first-stage uncertainty via paired cluster jackknife.
3. Run an **adversarial referee audit** (testing every survivor for survivor/composition bias, post-treatment selection, `WPU01` deflation artifacts, FY2020 Addendum-B breaks, `2026-08-03`/`2026-08-26` cutoff sensitivity, and single-state/top-1% leverage).
4. Write a concise, decision-ready synthesis report to `markdowns/bite_rescue_power_report.md` classifying each candidate **R1–R6** as **Tier 1 (High-Power Promotable)**, **Tier 2 (Moderate-Power Improvement)**, **Diagnostics-Only**, or **Killed (with exact Phase 1 failure numbers)** — and explicitly state whether any candidate justifies a follow-up promotion into `code/designs/` and `draft/draft1.tex`.

---

## 6. Copy-Paste Delegation Prompt for the Next Gemini Neon Instance

```text
You are running a strictly isolated, outcome-blind statistical-power investigation in `/home/ken/Dropbox/projects/H-2A_Paper` following `markdowns/bite_rescue_power_plan.md`.

First, read `markdowns/bite_rescue_power_plan.md` and `AGENTS.md`, then run `git status --short`, `python scripts/agent_docs.py verify`, and `python scripts/agent_docs.py snapshot --scope code/designs/bite` and `--scope code/designs/policy_2025`, and read `content/designs/bite.md`, `content/designs/policy-2025.md`, `markdowns/identification_search_report.md`, and `outputs/reports/bite_design.md`.

Execute the full two-phase protocol in `markdowns/bite_rescue_power_plan.md`:
1. Isolation & rules: Write ALL scripts/intermediates strictly under `explorations/bite_power_rescue/` and the final report to `markdowns/bite_rescue_power_report.md`. Do NOT modify `code/c01_clean/`, `code/c02_build/`, `code/designs/*`, `code/welfare/`, `scripts/*`, `draft/*`, `content/*`, `agent/assumptions.toml`, or any retained `outputs/*`. Use cached local data only (no non-public NASS, no web/court lookups, no live API calls; use unified `county_year_panel.parquet` and `state_real_minwages.parquet`, never retired `fred_state_minwages.parquet`). Keep geo IDs as strings (`GEO-001`) and canonical 2009 AEWR null (`AEWR-001`). Never downgrade clustering below the treatment-assignment level (`aewr_region_id` G=17 for FLS; `state_fips` G=46 for 2025 rule) or skip CV3J jackknife / empirical size validation.
2. Phase 1 (strictly outcome-blind for FY2026 quantities; placebo/FE-only calibration for 2011-2025): Audit all 6 candidates (R1: effective min-wage-truncated 2025 floor & sub-state tiers; R2: direct-grower-only + CDL×Year/NOAA weather covariates & SDID/IFE/ANCOVA variance absorption; R3: fixed-cohort balanced direct-grower business panel & within-multi-state-employer cross-state reallocation; R4: contract-duration amortization & farm labor-cost-share bite; R5: 2025 local-wage crossover bite gated on offered-wage first stage F>=10, |Δπ̂|>=0.25; R6: stacked 2009-10 OES reversal + 2025 statewide cut on positions). Record exact G_eff, SD(X̃), first-stage π̂/F, state-year shared-shock ratio, empirical size at α=0.05 (500-999 placebo draws), and size-valid MDEs in `explorations/bite_power_rescue/phase1_triage.csv`.
3. Relaxed power gate (strict credibility): Advance a candidate to Phase 2 ONLY if it passes empirical size [0.03, 0.075], clean pre-policy placebos (p>0.10; flag 0.05<p<=0.10 as conditional), first-stage F>=10 & |π̂|>=0.25 (where applicable), AND materially beats current benchmarks (aggregate/effective-floor statutory hours MDE<=0.60 and ε_cost MDE<=1.20; direct-grower/firm/reallocation hours MDE<=0.45 / positions MDE<=0.35; differential 2-SD spread MDE<=0.60 or 1a-PPML MDE<=1.00). If ZERO candidates pass, STOP without running noisy Phase 2 regressions and document the impossibility result cleanly.
4. Phase 2 (survivors only): Lock `explorations/bite_power_rescue/phase2_registry.json` + SHA-256 BEFORE estimating post-policy outcomes, classify survivors into Tier 1 (MDE<=0.40) vs Tier 2 (0.40<MDE<=0.60), and report exact CV3J CIs, matched pre-policy placebos, Holm-adjusted p-values, LOO influence, and an adversarial referee critique in `markdowns/bite_rescue_power_report.md`.
```
