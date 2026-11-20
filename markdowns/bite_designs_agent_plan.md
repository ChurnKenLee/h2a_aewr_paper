# Plan: AEWR bite designs across the panel regression, panel IV and border designs

Hand-off brief for an agent. Read it in full before starting.

## 1. Goal

The AEWR varies only by AEWR region and year. After region and year effects,
that leaves about 220 region-year deviations with an SD near $0.40, which is
too little to estimate the average AEWR effect on H-2A use precisely. A **bite**
measure adds county-level variation: how far the AEWR sits above what local
domestic labor costs.

Explore whether a predetermined bite, interacted with the AEWR, gives credible
and better-powered estimates in each of the three existing designs:

1. the plain panel regression (two-way fixed-effects OLS of outcomes on the real AEWR);
2. the panel IV (`code/designs/panel_iv`);
3. the border discontinuity (`code/designs/border_discontinuity`).

This is exploration. Do not change any retained specification, table or
pipeline stage. Deliver a report and scratch scripts, and propose
integration separately for the user to approve.

## 2. Read first

- `AGENTS.md` (repository rules). Run `python scripts/agent_docs.py snapshot --scope code/designs`
  and read what it names for the three design branches and `code/designs/mundlak_chamberlain`.
- `content/designs/panel-iv.md`, `content/designs/border-discontinuity.md`,
  `content/designs/mundlak-chamberlain.md` (the treatment registry section
  defines the frozen bite and fraction affected), `content/designs/difference-in-differences.md`.
- `markdowns/bite_designs_exploration.md`: all prior bite results, including
  failed designs. Section 1b is the starting point.
- `markdowns/aewr_identification_notes.html`: where the AEWR variation lives,
  and the dissimilar-donor argument for the panel IV.
- `outputs/reports/border_redesign.md`: the border design's own power and
  feasibility verdict.

## 3. What is already known

**Variation:**

- Year effects explain 74% of real-AEWR variance and region effects 22%. The
  residual SD is $0.46 across 238 region-years, or $0.37 weighting counties
  equally.
- After county and year effects, only 4–9% of outcome variance is at the
  region-year level.

**Panel IV (column 4):**

- First stage: coefficient 0.170, F 44.7, partial R² 0.106. IV SEs are 1.8–3.6
  times OLS.
- Dissimilar donors move much less with the target than other same-region
  subregions do (wages 0.12 against 0.28; outcomes net of the AEWR about 0
  against 0.24–0.30). The exclusion argument is defensible; precision is the
  problem.

**Plain panel OLS, certified workers per 2011 farm worker:** −0.040 (0.025).
A lead of the AEWR is about 60% of the current coefficient.

**Bites that failed:**

- 2007 Census pay per hired worker mostly measures employment duration.
- Employer bindingness (paying exactly the AEWR in a baseline year) is
  transient, and its result does not replicate across cohorts.
- 94% of H-2A hourly hours are paid exactly the AEWR. Bindingness on H-2A
  offers is therefore near-universal and is **not** the right validation target
  for a bite. The bite is about the H-2A-versus-domestic labor cost margin.

**The bite that works (section 1b of the exploration report):**

- Definition: frozen gap = real AEWR − max(ACS CZ p25 real hourly wage, real
  applicable agricultural minimum wage), averaged over 2008 and 2010. It is
  centered within region and scaled to unit SD (within-region SD $1.03; 62% of
  variance within region).
- 1(b) (E × AEWR with county and region×year FE): certified share 0.0025
  (0.0023), a precise null with a flat 2008–2010 pre-period.
- 1(a) (adds E × year FE): uninformative.
- Script: `explorations/bite/county_bite_acs.R`. Every earlier exploration
  script is in `explorations/bite/`; its README maps scripts to report sections.

**Already in the repo:**

- `aewr_cz_p25` and `aewr_cz_p25_l1` (contemporaneous, used by DiD) and the
  Mundlak–Chamberlain frozen bite families (`bite_f0809`, `bite_f0810`,
  `exposure_log_f0809/f0810`).
- MC results are in `outputs/tables/mc_v4_constructed_estimands.csv`. For
  example, `bite_f0809` with county and state×year FE gives any-application
  −0.24 (0.10) per dollar of bite.
- Reconcile the new estimates with these before reporting new numbers.

## 4. Common exposure construction (step 1, shared by all three designs)

Build one exposure table at the county level and reuse it. Keep geographic IDs
as strings and use `code/c00_shared/geography.R` normalizers.

| Name | Definition | Purpose |
|---|---|---|
| `E_gap` | frozen 2008/2010 gap above, within-region centered, unit SD | primary |
| `E_frac` | frozen share of p10–p90 point masses below the AEWR after the floor | MC-consistent alternative |
| `E_gap_early` | same gap using the earliest available ACS years (check `data/intermediate/acs_czone_wage_quantiles.parquet` for 2005–2007) | avoids 2008–2010 contamination by the 2009–10 rule reversal |
| `E_gap_ag` | gap against a farm-specific local wage, if one exists at county or CZ level (ACS agricultural workers; OEWS farmworker SOCs) | tests whether all-worker p25 proxies the farm outside option |
| `E_rank` | within-region rank of `E_gap` | robustness to outliers and ACS bunching |

**Checks on the exposure:**

- Distribution, within-region share of variance, and the correlation between
  the variants.
- How many distinct CZ values there are (ACS p25 bunches at round wages: 324
  distinct values across 608 CZs in 2015).
- Relevance in levels within region in 2011: any use, certified share and
  applications.
- How much exposure varies within border county pairs (step 5C depends on it).

Do not use contemporaneous or lagged time-varying gaps as the exposure.
The MC design page excludes them because local wages respond to the AEWR and
to local demand. Report them only as a labeled comparison with the DiD
branch.

## 5. Design-specific tasks

Use the retained outcome registry
(`code/designs/panel_iv/estimation_spec.json`). The primary outcome is
`h2a_cert_share_farm_workers_2011_start_year`. Also run certified hours,
applications and linked worksite businesses per 2011 farm worker. Use the
real AEWR (`aewr_ppi`). The window is 2011–2023 with BEA controls, and
2011–2024 with the continuing-controls set.

### A. Plain panel regression

1. Reproduce the baseline: y ~ A + controls | county + year (OLS for certified
   workers about −0.040).
2. **Heterogeneity, keeping the average effect:** y ~ A + A × E + controls |
   county + year. This reports how much the AEWR effect varies with bite
   while keeping region-level identification of the average effect.
3. **Bite-only identification:** 1(b) y ~ A × E | county + region×year; 1(a)
   adds E × year FE. Also report an event study i(year, E, ref = 2011) over
   2008–2024. Mark 2009–2010 as reversal-affected years.
4. **Convert the differential to the average effect.** Write a short model in
   which the AEWR effect scales with the bite. For example, the effect on
   H-2A demand is proportional to the AEWR's cost premium over the local
   floor, so β_c = β̄ · gap_c / mean(gap). Derive what the 1(b) coefficient
   implies for β̄ and its confidence interval. State the assumption plainly; it
   carries the conversion.

### B. Panel IV

1. Reproduce column 4: first stage 0.170, F 44.7.
2. **Bite-interacted IV without region×year FE:** endogenous A and A × E;
   instruments Z and Z × E (Z = `z_wage_seasonal_composition_real`). Report
   Kleibergen–Paap and Sanderson–Windmeijer conditional Fs and
   Anderson–Rubin sets.
3. **Bite-interacted IV with region×year FE:** endogenous A × E, instrument
   Z × E. The region-year AEWR main effect is absorbed. Z varies by target
   subregion within region-year, so Z × E keeps within-region variation.
   - The exclusion argument is the dissimilar-donor one, applied to the
     differential response.
   - Check the first stage, and whether IV and OLS differ (step A3). If they
     don't, the IV adds little.
4. **Co-movement check by exposure:** rerun the dissimilar-donor
   co-movement test (`explorations/bite/common_shocks.R`) separately for high-
   and low-exposure target subregions.

### C. Border discontinuity

The border design compares counties on either side of an AEWR-region border.
It uses county and border-component×year FE, clusters by region pair, and
uses WCR11-FNW wild bootstrap inference. Its policy window was revised to
2011–2025 on 2026-10-26 (see `design.json`). Its report calls the design
infeasible: 25 edges among 16 regions, with a within-component treatment SD of
$0.227.

1. **Exposure within pairs.** Measure how much exposure differs between the
   two sides of each pair. Border counties often share a CZ, so the frozen
   local floor is often identical, and the within-pair gap difference then
   equals the AEWR difference.
2. **Heterogeneity across pairs:** y ~ A + A × E_pair | county +
   component×year, where E_pair is the pair's mean baseline exposure. It asks
   whether border AEWR differences matter more where the AEWR bites harder.
3. **Gap as treatment:** use the county frozen-floor gap G_ct = A_rt − floor_c
   as the treatment within component-year. Show how much of its within-
   component variation comes from AEWR differences versus floor differences.
   Only the AEWR part is policy variation.
4. **Inference:** reuse the branch's region-pair wild bootstrap and its signed
   MDE machinery. Read `08_power_and_report.R` before writing new inference.

## 6. Inference, diagnostics and decision rules (all designs)

- **Inference:** report CRV1 by state, by AEWR region, and the panel IV
  cluster (`aewr_iv_cluster_id`). Add a restricted Webb wild cluster
  bootstrap over the 17 regions for every headline coefficient.
  `fwildclusterboot::boottest` failed earlier on these models; a hand-rolled
  FWL bootstrap works (see `explorations/bite/employer_robust.R`).
- **Pre-trends:** event studies with 2008, 2009 and 2010 as pre-years. 2009 has
  no canonical AEWR and 2009–2010 were under the 2008 rule, so treat them as
  contaminated pre-years and say so.
- **Placebos:**
  - E × next-year AEWR (lead).
  - Exposure built from a randomly permuted within-region assignment
    (500 permutations) to get a randomization distribution.
- **Power:** report the MDE (2.8 × SE) next to every estimate, in outcome units
  and as a percentage of the outcome mean.
- **Decision rule for "promising":** the design passes the pre-trend and
  lead placebos, survives the wild bootstrap and at least two exposure
  variants, and has an MDE below 25% of the outcome mean. Report every
  specification run, including failures. Do not select on significance.

## 7. Known data pitfalls

- 2009 AEWR is missing (canonical source gap); 2010 lags are missing.
- From 2023 the AEWR is occupation-specific. `cert_hours_at_aewr` is zero from
  2023 in the shared panel.
- Employer counts (and the linked-businesses outcome) break in 2020, when
  businesses named on association and contractor filings appear (about 10,000
  to 36,000 employers).
- County H-2A counts are fractional where multi-county cases are allocated. 38%
  of county-year worker counts are non-integer, so "any use" is inflated.
- ACS quantiles are commuting-zone level and PPI-deflated (`wage_p*` are
  real). County `wage_p*` values repeat within CZ.
- BEA-based controls end in 2023; use the continuing-controls set for 2024.

## 8. Constraints

- No commits. No source acquisition, network calls or credentialed steps.
  Don't edit retained outputs, `data/raw` or crosswalks.
- New scripts go in `explorations/bite/` (or a sibling folder), following its
  README conventions. Outputs go to a scratch directory outside the repo.
  Write the final report to `markdowns/bite_designs_cross_design.md`. Every
  number in it must trace to a script named in the report.
- R: `R_LIBS_USER=$(ls -d $PWD/renv/library/*/R-*/* | head -1) Rscript --vanilla <script>`,
  then `git checkout -- Rplots.pdf`. Python: `UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync`.
- If any result suggests changing a design's specification, write it as a
  proposal in the report (what changes, which README or design page to update).
  Do not implement it.

## 9. Deliverable

`markdowns/bite_designs_cross_design.md`, with:

1. the exposure table and its diagnostics;
2. one section per design, with specification, estimates, MDEs, pre-trends,
   placebos and bootstrap p-values;
3. reconciliation with the Mundlak–Chamberlain bite outputs;
4. the model converting bite heterogeneity into a bound on the average AEWR
   effect, with its assumptions;
5. a recommendation for each design (pursue, report as supporting evidence, or
   drop), plus proposed integration steps for the user to approve.
